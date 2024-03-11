from __future__ import annotations

from contextlib import contextmanager
import math
import os
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from transformers import PreTrainedTokenizer

from vectorlm.dataset import Dataset
from vectorlm.utils.data_utils import Config
from vectorlm.utils.save_utils import (
    checkpoint_exists,
    get_latest_checkpoint_dir,
    load_metadata,
    load_model,
    load_optimizer,
    load_scheduler,
    save_metadata,
    save_model,
    save_optimizer,
    save_scheduler,
)


@contextmanager
def timer_placeholder(task_name: str):
    try:
        yield  # start code block
    finally:
        # run before exiting
        return


class Trainer:
    """Main trainer class.

    Attributes
    ----------
        config: A training config.
        gas: An integer number of gradient accumulation steps.
        model: A model we are training.
        tokenizer: A model's tokenizer.
        optimizer: An optimizer we are using.
        lr_scheduler: An LR scheduler for the optimizer.
        dataset: A `Dataset` class.
        tr_step: An integer training step.
        metric: A metric we are tracking for model training.
        wandb_logging: A boolean for whether we are logging using wandb.
        logging_steps: An integer for how often we log.
        num_update_steps_per_epoch: An integer number of training steps per
            epoch.
        max_steps: An integer maximum number of training steps for this run.
        saving_steps: An integer for how often we save.

    """

    def __init__(
        self,
        config: Config,
        enable_wandb_logging: bool,
        original_dataset_length: int,
        timer_handle=timer_placeholder,
    ) -> None:
        """Initialize the Trainer class.

        Args:
        ----
            config: The training config.
            enable_wandb_logging: Whether to enable wandb logging.
            original_dataset_length: The length of the original dataset
                (divided by the batch size).
            timer_handle: Optional context manager for profiling.
        """
        self.config = config
        self.gas = config.gradient_accumulation_steps
        if self.gas < 1:
            msg = "Gradient accumulation steps need to be >=1."
            raise ValueError(msg)
        self.tr_step = 0
        self.metric = math.inf
        self.wandb_logging = enable_wandb_logging
        self.logging_steps = self.config.logging_steps
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataset = None
        self.num_update_steps_per_epoch = None
        self.max_steps = None
        self.saving_steps = None
        self.timer_handle = timer_handle
        self._post_process(original_dataset_length)

    def _post_process(self, ds_orig_length: int) -> None:
        """Calculate steps for weight updates and saving."""
        sharded_ds_orig_len = math.ceil(
            ds_orig_length / dist.get_world_size(),
        )
        self.num_update_steps_per_epoch = max(
            sharded_ds_orig_len // self.gas, 1,
        )
        self.max_steps = math.ceil(
            self.config.epochs * self.num_update_steps_per_epoch,
        )
        self.saving_steps = int(
            self.config.save_frequency * sharded_ds_orig_len,
        )

    def prepare_trainer(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | ReduceLROnPlateau,
    ) -> None:
        """Set all essential training requirements.

        Args:
        ----
            model: The sharded model.
            tokenizer: The model's tokenizer.
            dataset: The `Dataset` class.
            optimizer: The training optimizer.
            lr_scheduler: The LR scheduler.

        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def save_checkpoint(self, epoch: int) -> None:
        """Save all states.

        Args:
        ----
            epoch: The current training epoch.

        """
        rank = dist.get_rank()
        gathered_processed_ids = _gather(
            self.dataset.get_processed_ids(),
        )
        meta_dict = {
            "tr_step": self.tr_step + 1,
            "processed_ids": gathered_processed_ids,
            "epoch": epoch,
        }
        save_dir = os.path.join(
            self.config.output_dir,
            "checkpoints",
            f"epoch_{epoch}",
            f"checkpoint_{self.tr_step}",
        )
        if rank == 0:
            save_metadata(save_dir, meta_dict)

        with self.timer_handle("trainer_save_model"):
            save_model(self.model, save_dir, rank)

        with self.timer_handle("trainer_save_optimizer"):
            save_optimizer(self.optimizer, self.model, save_dir, rank)

        with self.timer_handle("train_save_scheduler"):
            save_scheduler(self.lr_scheduler, save_dir, rank)

        dist.barrier()

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load all states.

        Args:
        ----
            checkpoint_dir: The directory under which all checkpoints are
                saved.

        Returns:
        -------
            The checkpointed epoch to be used by the outer loop.

        """
        rank = dist.get_rank()
        step, epoch, ids = load_metadata(checkpoint_dir)
        self.tr_step = step
        self.dataset.set_processed_ids(ids)
        self.dataset.setup_dataloaders()
        load_model(self.model, checkpoint_dir, rank)
        load_optimizer(self.optimizer, self.model, checkpoint_dir, rank)
        load_scheduler(self.lr_scheduler, checkpoint_dir, rank)
        dist.barrier()
        return epoch

    def find_checkpoint(self, checkpoint_dir: str) -> int:
        """Find and load checkpoint if it exists.

        Args:
        ----
            checkpoint_dir: The checkpointing directory.

        Returns:
        -------
            The checkpointed epoch. If no checkpoint exists, it returns a
            default value of 0.

        """
        checkpoint = checkpoint_exists(checkpoint_dir)
        if checkpoint:
            main_ckpt_dir = os.path.join(checkpoint_dir, "checkpoints")
            latest_ckpt_dir = get_latest_checkpoint_dir(main_ckpt_dir)
            full_ckpt_dir = os.path.join(main_ckpt_dir, latest_ckpt_dir)
            print_main(f"Checkpoint found at {full_ckpt_dir}")
            checkpointed_epoch = self.load_checkpoint(full_ckpt_dir)
        else:
            self.dataset.setup_dataloaders()
            checkpointed_epoch = 0
        return checkpointed_epoch

    def step(
        self,
        train_batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[float, float | None]:
        """Step in an all-encapsulating manner.

        Steps training, and if required, evals and checkpoints.

        Args:
        ----
            train_batch: The training batch.
            epoch: The current training epoch.

        """
        if (
            self.config.checkpointing_enabled
        ) and (
            (self.tr_step + 1) % self.saving_steps == 0
        ):
                self.save_checkpoint(epoch)

        num_tokens = len(train_batch["input_ids"].flatten())
        with self.timer_handle("train_step", {"num_tokens": num_tokens}):
            train_loss = self.train_step(train_batch, epoch)

        test_loss = None
        if self.tr_step % self.logging_steps == 0:
            test_loss = self.eval_step(epoch)
        self.tr_step += 1
        return train_loss, test_loss


    def train_step(self, batch: dict[str, torch.Tensor], epoch: int) -> float:
        """Step training once.

        Args:
        ----
            batch: The training batch.
            epoch: The current training epoch.

        """
        ids = batch.pop("id").to(torch.cuda.current_device())
        batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
        batch["labels"] = batch["labels"].type(torch.LongTensor)
        batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}
        self.dataset.update_processed_ids(ids)

        if (self.tr_step + 1) % self.gas != self.gas - 1:
            if hasattr(self.model, "no_sync"):
                # fsdp: no need to sync while accumulating gradients
                with self.model.no_sync():
                    out = self.model(**batch)
                    tr_step_loss = out.loss
                    (tr_step_loss / self.gas).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
            else:
                # non-fsdp
                out = self.model(**batch)
                tr_step_loss = out.loss
                (tr_step_loss / self.gas).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

        else:
            # next forward / backward pass will be synced
            dist.barrier()
            out = self.model(**batch)
            tr_step_loss = out.loss
            (tr_step_loss / self.gas).backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.metric)
            else:
                self.lr_scheduler.step()
            print_main(f"LR: {self.lr_scheduler.get_last_lr()[0]}")
            self.optimizer.zero_grad()
        gathered_tr_step_loss = _gather(tr_step_loss.reshape(1)).mean().item()

        if self.wandb_logging:
            self.log(gathered_tr_step_loss, epoch, "train")

        return gathered_tr_step_loss

    def eval_step(self, epoch: int) -> float:
        """Run evaluation.

        Args:
        ----
            epoch: The current training epoch.

        """
        print_main("Evaluating")
        self.model.eval()
        eval_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        for _, batch in enumerate(self.dataset.eval_dataloader):
            with torch.no_grad():
                batch.pop("id")
                batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
                num_tokens = len(batch["input_ids"].flatten())
                batch["labels"] = batch["labels"].type(torch.LongTensor)
                batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

                with self.timer_handle("eval_step", {"num_tokens": num_tokens}):
                    out = self.model(**batch)
                    eval_loss += out.loss

        gathered_eval_loss = _gather(eval_loss.reshape(1)).mean().item()
        mean_eval_loss = gathered_eval_loss / len(self.dataset.eval_dataloader)

        self.metric = gathered_eval_loss

        print_main(f"Step: {self.tr_step}, eval loss: {mean_eval_loss}")
        if self.wandb_logging:
            self.log(mean_eval_loss, epoch, "eval")

        self.model.train()
        return gathered_eval_loss

    def log(self, loss: float, epoch: int, mode: str = "train") -> None:
        """Log values.

        Args:
        ----
            loss: The loss being logged.
            epoch: The current training epoch.
            mode: One of `train` or `eval`.

        """
        if mode not in {"train", "eval"}:
            msg = "`mode` argument needs to be 'train' or 'eval'."
            raise ValueError(msg)

        num_tokens_processed = (
            dist.get_world_size()
            * (self.tr_step + 1 + self.dataset.original_length * epoch)
            * self.dataset.train_bs
            * self.tokenizer.model_max_length
        ) / 1e6

        if dist.get_rank() == 0:
            if mode == "train":
                commit = False
                if self.tr_step % self.config.logging_steps != 0:
                    commit = True
                wandb.log(
                    {
                        "train/step_loss": loss,
                        "millions_of_tokens": num_tokens_processed,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    },
                    commit=commit,
                    step=self.tr_step + (epoch * self.dataset.original_length),
                )
            else:
                wandb.log(
                    {
                        "test/loss": loss,
                        "millions_of_tokens": num_tokens_processed,
                    },
                    step=self.tr_step + (epoch * self.dataset.original_length),
                )


def _gather(x: torch.Tensor) -> torch.Tensor:
    output_tensors = [
        x.clone() for _ in range(dist.get_world_size())
    ]
    dist.all_gather(output_tensors, x)
    return torch.cat(output_tensors, dim=0)


def print_main(*args: list[Any], **kwargs: dict[str, Any]) -> None:
    """Print only on the main rank."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)
