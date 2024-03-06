# Renamed from examples/llama_example.py
import argparse
import math
import os
import sys
from argparse import Namespace

import torch
import torch.distributed as dist
from torch.optim import AdamW
from tqdm import tqdm
from transformers import set_seed
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from vectorlm.dataset import Dataset
from vectorlm.trainer import Trainer
from vectorlm.utils.data_utils import Config
from vectorlm.utils.misc_utils import cleanup, setup, wandb_setup
from vectorlm.utils.model_utils import (
    hook_activation_checkpointing,
    initialize_lora_model_and_tokenizer,
    shard_model,
)
from vectorlm.utils.optimizer_utils import get_custom_scheduler
from vectorlm.utils.save_utils import save_consolidated_model


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns
    -------
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        default="configs/config.yaml",
        required=False,
    )
    return parser.parse_args()


def main(config: Config) -> None:
    """Define the main calling function."""
    training_args = config.train_parameters

    # set a seed
    set_seed(training_args.seed)

    # set CUDA related dependencies
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Rank: {rank}, World size: {world_size}")
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()

    # setup wandb
    if rank == 0:
        wandb_setup(config, **config.wandb_config)
    dist.barrier()

    # load model and tokenizer
    state_dict_path = getattr(config, "state_dict", None)

    model, tokenizer = initialize_lora_model_and_tokenizer(
        config.model,
        training_args.use_mp,
        training_args.use_flash_attention,
        training_args.max_seq_len,
        config.lora_peft_config,
    )

    model = shard_model(
        model,
        LlamaDecoderLayer,
        training_args.use_mp,
        training_args.use_activation_checkpointing,
        training_args.sharding_strategy,
    )

    if training_args.use_activation_checkpointing:
        hook_activation_checkpointing(model, LlamaDecoderLayer)

    # load dataset
    dataset = Dataset(
        config=config.dataset,
        tokenizer=tokenizer,
    )

    # instantiate trainer
    trainer = Trainer(
        config=training_args,
        enable_wandb_logging=config.enable_wandb_logging,
        original_dataset_length=dataset.original_length,
    )

    # load optimizer
    optimizer = AdamW(
        model.parameters(),
        **training_args.optimizer,
    )

    # load lr scheduler
    lr_scheduler = get_custom_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        math.ceil(
            trainer.num_update_steps_per_epoch * training_args.warmup_ratio,
        ),
        trainer.max_steps,
    )

    trainer.prepare_trainer(
        model,
        tokenizer,
        dataset,
        optimizer,
        lr_scheduler,
    )

    # TODO: support restoring LoRA fine-tuning
    trainer.dataset.setup_dataloaders()
    checkpointed_epoch = 0

    for epoch in range(checkpointed_epoch, training_args.epochs):
        trainer.model.train()
        train_dl_iterator = iter(dataset.train_dataloader)
        for _ in tqdm(
            range(len(dataset.train_dataloader)),
            disable=rank != 0,
            file=sys.__stdout__,
        ):
            batch = next(train_dl_iterator)
            trainer.step(batch, epoch)

        if epoch == training_args.epochs - 1:
            hf_save_dir = os.path.join(training_args.output_dir, "final-model")
        else:
            hf_save_dir = os.path.join(
                training_args.output_dir,
                "checkpoints",
                f"epoch_{epoch}",
                "end-epoch-model",
            )
        save_consolidated_model(trainer.model, hf_save_dir, rank)
        if rank == 0:
            tokenizer.save_pretrained(hf_save_dir)

        dataset.reset_dataloaders()


if __name__ == "__main__":
    args = parse_args()
    config = Config(yaml_path=args.yaml_path)
    setup(config.train_parameters.output_dir)
    main(config)
    cleanup()
