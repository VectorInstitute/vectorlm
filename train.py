from transformers import set_seed, get_scheduler
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
import torch
import wandb
import datasets
import argparse
import math
import os
from tqdm import tqdm
import sys
from typing import List, Dict, Any, Callable
import re
from copy import deepcopy
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import torch.distributed

from data_utils import DataCollatorWithPadding, Config, pack_examples, get_mdpi_mtb_dataloaders, reset_mdpi_mtb_dataloader
from save_utils import get_latest_checkpoint_dir, load_metadata, checkpoint_exists, gather, load_model, load_optimizer, save_model, save_consolidated_model, save_optimizer, save_metadata
from model_utils import PlateaeuWithWarmup, load_model_and_tokenizer, fsdp_config, apply_activation_checkpointing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--yaml_path", default="configs/config.yaml", required=False
    )
    return parser.parse_args()


def print_rank0(x: str, rank: int):
    """Less verbose printing for main rank"""
    if rank == 0:
        print(x)


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def wandb_setup(config: Config, training_args: Config):
    os.makedirs(
        os.path.join(training_args.output_dir, "final-model"), exist_ok=True
    )

    wandb_dict = config.train.wandb_config.to_dict()
    wandb.init(
        config=config.train.hf_training_arguments.to_dict(),
        **wandb_dict,
    )
    wandb.define_metric("millions_of_tokens")
    wandb.define_metric("train/*", step_metric="millions_of_tokens")
    wandb.define_metric("test/*", step_metric="millions_of_tokens")


def train_function(config):
    training_args = config.train.hf_training_arguments
    set_seed(training_args.seed)

    local_rank: int = int(os.environ["LOCAL_RANK"])
    rank: int = int(os.environ["RANK"])
    world_size: int = int(os.environ["WORLD_SIZE"])
    printm: Callable = partial(print_rank0, rank=rank)
    print(f"Rank: {rank}, World size: {world_size}")
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()

    if rank == 0:
        wandb_setup(config, training_args)
    dist.barrier()

    checkpoint: bool = checkpoint_exists(training_args.output_dir)
    checkpointed_step: int = 0
    checkpointed_epoch: int = 0
    if checkpoint:
        printm("Checkpoint found")
    
    to_remove: List[int] = []

    model, tokenizer = load_model_and_tokenizer(config)
    # breakpoint()

    if checkpoint:
        checkpointed_step, checkpointed_epoch, to_remove = load_metadata(
            training_args.output_dir,
            local_rank,
        )
    dist.barrier()

    orig_length, train_dataloader, eval_dataloader = get_mdpi_mtb_dataloaders(
        tokenizer,
        config,
        to_remove,
    )

    fsdp_cfg: Dict[str, Any] = fsdp_config()
    model = FSDP(model, **fsdp_cfg)

    printm(
        "Train/Eval dataloader sizes after sharding are "
        f"{len(train_dataloader)}, {len(eval_dataloader)}"
    )
    print("Per device model parameters are "
          f"{sum(p.numel() for p in model.parameters())}"
    )

    gradient_accumulation_steps: int = \
        training_args.gradient_accumulation_steps

    orig_length: int = math.ceil(orig_length / world_size)

    num_update_steps_per_epoch: int = max(
        orig_length // gradient_accumulation_steps, 1
    )

    max_steps: int = \
        math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    logging_steps: int = training_args.logging_steps
    saving_steps: int = int(training_args.save_frequency * orig_length)

    optimizer: AdamW = AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
    )

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        math.ceil(max_steps * training_args.warmup_ratio),
        max_steps,
    )

    printm("Optimizers and LR scheduler created")
    printm(
        f"Original dataloader length: {orig_length * world_size}, "
        f"per device *update* steps: {max_steps}, warmup steps "
        f"{math.ceil(max_steps * training_args.warmup_ratio)}, " 
        f"training for {training_args.num_train_epochs} epochs "
        f"which is a total of "
        f"{orig_length * training_args.num_train_epochs} "
        f"per device iterations"
    )

    if checkpoint:
        latest_checkpoint_folder = get_latest_checkpoint_dir(
            os.path.join(training_args.output_dir, "checkpoints")
        )
        save_dir = os.path.join(
            training_args.output_dir,
            "checkpoints",
            latest_checkpoint_folder,
        )
        load_model(model, save_dir, rank)
        load_optimizer(optimizer, model, save_dir, rank)
        dist.barrier()
        printm("All states restored")

    for epoch in range(checkpointed_epoch, training_args.num_train_epochs):
        model.train()

        if checkpoint:
            processed_ids = torch.tensor(to_remove).to(torch.cuda.current_device())
        else:
            processed_ids = torch.tensor([]).to(torch.cuda.current_device())

        train_dl_iterator = iter(train_dataloader)

        for step in tqdm(
            range(len(train_dataloader)),
            disable=rank != 0,
            file=sys.__stdout__,
        ):
            tr_step = checkpointed_step + step
            batch = next(train_dl_iterator)

            # saving
            if (
                (tr_step + 1) % saving_steps == 0
            ) and training_args.checkpointing_enabled:
                gathered_processed_ids = gather(processed_ids)
                meta_dict = {
                    "tr_step": tr_step + 1,
                    "processed_ids": gathered_processed_ids,
                    "epoch": epoch,
                }
                save_dir = os.path.join(
                    training_args.output_dir,
                    "checkpoints",
                    f"epoch_{epoch}",
                    f"checkpoint_{tr_step}",
                )
                save_model(model, save_dir, rank)
                save_optimizer(optimizer, model, save_dir, rank)
                save_metadata(
                    training_args.output_dir, meta_dict, tr_step, epoch
                )

            # training
            ids = batch.pop("id").to(torch.cuda.current_device())
            batch.pop("raw_data_id") if "raw_data_id" in batch else None
            batch['input_ids'] = batch['input_ids'].type(torch.LongTensor)
            batch['labels'] = batch['labels'].type(torch.LongTensor)
            processed_ids = torch.cat([processed_ids, ids])
            if (
                tr_step + 1
            ) % gradient_accumulation_steps != gradient_accumulation_steps - 1:
                # no need to sync while accumulating gradients
                with model.no_sync():
                    out = model(**batch)
                    tr_step_loss = out.loss
                    (tr_step_loss / gradient_accumulation_steps).backward()
                    model.clip_grad_norm_(training_args.max_grad_norm)
            else:
                # next forward / backward pass will be synced
                dist.barrier()
                out = model(**batch)
                tr_step_loss = out.loss
                (tr_step_loss / gradient_accumulation_steps).backward()
                model.clip_grad_norm_(training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                printm(f"LR: {lr_scheduler.get_last_lr()[0]}")
                optimizer.zero_grad()
            gathered_tr_step_loss = gather(tr_step_loss.reshape(1)).mean().item()

            # logging
            if rank == 0:
                num_tokens_processed = (
                    world_size
                    * (tr_step + 1 + orig_length * epoch)
                    * training_args.per_device_train_batch_size
                    * config.train.max_seq_len
                ) / 1e6
                commit = True if tr_step % logging_steps != 0 else False
                wandb.log(
                    {
                        "train/step_loss": gathered_tr_step_loss,
                        "millions_of_tokens": num_tokens_processed,
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    commit=commit,
                    step=tr_step + (epoch * orig_length),
                )

            # evaluating
            if tr_step % logging_steps == 0:
                printm("Evaluating")
                model.eval()
                eval_loss = torch.tensor(0.0).to(torch.cuda.current_device())
                for _, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        batch.pop("id")
                        batch.pop("raw_data_id") if "raw_data_id" in batch else None
                        batch['input_ids'] = batch['input_ids'].type(torch.LongTensor)
                        batch['labels'] = batch['labels'].type(torch.LongTensor)
                        out = model(**batch)
                        eval_loss += out.loss
                gathered_eval_loss = gather(eval_loss.reshape(1)).mean().item()
                if rank == 0:
                    printm(
                        f"Step: {tr_step}, train loss: {gathered_tr_step_loss}, eval loss: {gathered_eval_loss / len(eval_dataloader)}"
                    )
                    wandb.log(
                        {
                            "test/loss": gathered_eval_loss / len(eval_dataloader),
                            "millions_of_tokens": num_tokens_processed,
                        },
                        step=tr_step + (epoch * orig_length),
                    )
                model.train()

        if checkpoint:
            checkpointed_step = 0
            checkpoint = False
        
        if training_args.num_train_epochs > 1:
            # reset dataset (add back processed ids)
            train_dataloader = reset_mdpi_mtb_dataloader(tokenizer, config)
    
    save_consolidated_model(model, training_args.output_dir, rank)

if __name__ == '__main__':
    args = parse_args()
    config = Config(yaml_path=args.yaml_path)
    config.train.hf_training_arguments["output_dir"] = args.output_dir
    setup()
    train_function(config)
    cleanup()