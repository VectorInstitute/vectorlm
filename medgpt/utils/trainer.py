from .data_utils import Config, Dataset
from .model_utils import load_model_and_tokenizer, fsdp_config, hook_activation_checkpointing
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import torch
import math
import torch.distributed as dist
import wandb
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP
)

class Trainer:
    """Main trainer class."""

    def __init__(self,
            config: Config,
            model_tokenizer_path: str,
            enable_wandb_logging: bool,
        ) -> None:
        self.config = config
        self.gas = config.gradient_accumulation_steps
        if self.gas < 1:
            raise ValueError("Gradient accumulation steps need to be >=1.")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_tokenizer_path,
            config.use_mp,
            config.use_flash_attention,
            config.max_seq_len,
        )
        self.optimizer = None
        self.lr_scheduler = None
        self.dataset = None
        self.tr_step = 0
        self.metric = math.inf
        self.wandb_logging = enable_wandb_logging
        self.logging_steps = self.config.logging_steps
        self.num_update_steps_per_epoch = None
        self.max_steps = None
        self.saving_steps = None
    
    @staticmethod
    def print_main(*args, **kwargs) -> None:
        """Print only on the main rank."""
        if dist.is_initialized:
            if dist.get_rank() == 0:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)
    
    def _post_process(self) -> None:
        """Calculate steps for weight updates and saving."""
        shard_ds_orig_len = math.ceil(
            self.dataset.original_length / dist.get_world_size()
        )
        num_update_steps_per_epoch: int = max(
            shard_ds_orig_len // self.gas, 1
        )
        self.max_steps = math.ceil(
            self.config.epochs * num_update_steps_per_epoch
        )
        self.saving_steps = int(self.config.save_frequency * shard_ds_orig_len)

    def set_optim_and_scheduler(
        self,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | ReduceLROnPlateau,
    ) -> None:
        """
        Set the optimizer and scheduler post-hoc because we can only define it
        after model parameters are created/sharded.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    def set_dataset(self, dataset: Dataset) -> None:
        """Set the dataset attribute."""
        self.dataset = dataset
        self._post_process()

    def shard_model(self, layer_to_wrap: torch.nn.Module) -> None:
        fsdp_cfg = fsdp_config(
            self.config.use_mp,
            layer_to_wrap,
        )
        self.model = FSDP(self.model, **fsdp_cfg)
        print(
            "Model sharded. Per device model parameters are ",
            f"{sum(p.numel() for p in self.model.parameters())}",
        )

        if self.config.use_activation_checkpointing:
            hook_activation_checkpointing(self.model, layer_to_wrap)


    def save_checkpoint(self) -> None:
        pass

    def load_checkpoint(self) -> None:
        pass

    def load_diff_model_weights(self, path: str) -> None:
        """
        Useful when going from pretraining to SFT. We only want to load in
        the model weights, nothing else.
        """
        pass

    def set_train_step(self, tr_step: int) -> None:
        self.tr_step = tr_step

    def train_step(self, batch: dict[str, torch.Tensor], epoch: int) -> float:
        ids = batch.pop("id").to(torch.cuda.current_device())
        batch.pop("raw_data_id") if "raw_data_id" in batch else None #TODO: remove
        batch['input_ids'] = batch['input_ids'].type(torch.LongTensor)
        batch['labels'] = batch['labels'].type(torch.LongTensor)
        self.dataset.update_processed_ids(ids)

        if (self.tr_step + 1) % self.gas != self.gas - 1:
            # no need to sync while accumulating gradients
            with self.model.no_sync():
                out = self.model(**batch)
                tr_step_loss = out.loss
                (tr_step_loss / self.gas).backward()
                self.model.clip_grad_norm_(self.config.max_grad_norm)
        else:
            # next forward / backward pass will be synced
            dist.barrier()
            out = self.model(**batch)
            tr_step_loss = out.loss
            (tr_step_loss / self.gas).backward()
            self.model.clip_grad_norm_(self.config.max_grad_norm)
            self.optimizer.step()
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.metric)
            else:
                self.lr_scheduler.step()
            self.print_main(f"LR: {self.lr_scheduler.get_last_lr()[0]}")
            self.optimizer.zero_grad()
        gathered_tr_step_loss = _gather(tr_step_loss.reshape(1)).mean().item()

        if self.wandb_logging:
            self.log(gathered_tr_step_loss, epoch, "train")

        return gathered_tr_step_loss
    
    def eval_step(self, epoch: int) -> float:
        self.print_main("Evaluating")
        self.model.eval()
        eval_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        for _, batch in enumerate(self.dataset.eval_dataloader):
            with torch.no_grad():
                batch.pop("id")
                batch.pop("raw_data_id") if "raw_data_id" in batch else None #TODO: remove
                batch['input_ids'] = batch['input_ids'].type(torch.LongTensor)
                batch['labels'] = batch['labels'].type(torch.LongTensor)
                out = self.model(**batch)
                eval_loss += out.loss
        gathered_eval_loss = _gather(eval_loss.reshape(1)).mean().item()
        mean_eval_loss = gathered_eval_loss / len(self.dataset.eval_dataloader)

        self.metric = gathered_eval_loss

        self.print_main(f"Step: {self.tr_step}, eval loss: {mean_eval_loss}")
        if self.wandb_logging:
            self.log(mean_eval_loss, epoch, "eval")

        self.model.train()

    def log(self, loss: float, epoch: int, mode: str = "train") -> None:
        if mode != "train" and mode != "eval":
            raise ValueError("`mode` argument needs to be 'train' or 'eval'.")

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