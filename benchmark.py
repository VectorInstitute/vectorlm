# Renamed from examples/llama_example.py
import argparse
import contextlib
import json
import math
import os
import sys
import time
from argparse import Namespace
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW
from tqdm import tqdm
from transformers import set_seed
from peft.utils.other import fsdp_auto_wrap_policy

from vectorlm.dataset import Dataset
from vectorlm.trainer import Trainer
from vectorlm.utils.data_utils import Config
from vectorlm.utils.misc_utils import cleanup, setup, wandb_setup
from vectorlm.utils.model_utils import (
    hook_activation_checkpointing,
    load_model_and_tokenizer,
    shard_model,
    get_submodule_by_pattern,
    get_lora_model_from_base_model,
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
    parser.add_argument(
        "--model_name",
        required=True,
    )
    return parser.parse_args()


# unix timestamp
launch_time = time.time()
os.makedirs("data/benchmark", exist_ok=True)
output_path = "data/benchmark/{}.jsonl".format(launch_time)


def write_metrics(metric_name: str, value: Optional[Any] = None) -> None:
    """
    Write metric and time elapsed to output file.
    Write to disk only if process rank is 0.

    Params:
        metric_name: string indicating type of metric
        value: JSON-serializable value,
            or None to log only time elapsed
    """
    time_since_launch = time.time() - launch_time
    output_dict = {
        "name": metric_name,
        "time_since_launch": time_since_launch,
        "value": value,
    }
    output_line = json.dumps(output_dict)

    if dist.get_rank() == 0:
        with open(output_path, "a") as output_file:
            output_file.write(output_line + "\n")


@contextlib.contextmanager
def track_time(task_name: str, extra_info: Dict[str, Any] = {}):
    start_time = time.time()
    try:
        yield
    finally:
        time_elapsed = time.time() - start_time
        write_metrics(task_name, {"time_elapsed": time_elapsed, **extra_info})


def get_device_info() -> Dict[str, str | int]:
    """
    Get CUDA info as a dict.

    Returns:
        Dict including device_name and world size
    """
    return dict(
        device_name=torch.cuda.get_device_name(),
        local_rank=int(os.environ["LOCAL_RANK"]),
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )


def get_is_flash_attention_supported() -> bool:
    """
    Returns:
        Whether Flash Attention is supported based on
        the given CUDA device capability.
    """
    version_major, _ = torch.cuda.get_device_capability()
    return version_major >= 8


def get_slurm_env() -> Dict[str, str]:
    """
    Returns a dictionary of all env var starting with "SLURM_".
    """
    output = {
        key: value for key, value in os.environ.items() if key.startswith("SLURM_")
    }
    return output


def main(config: Config, model_name: str) -> None:
    """Define the main calling function."""
    write_metrics("model_name", model_name)
    write_metrics("config", {**config.__dict__})
    write_metrics("device_info", get_device_info())
    write_metrics("slurm_info", get_slurm_env())

    training_args = config.train_parameters

    # set a seed
    set_seed(training_args.seed)

    # set CUDA related dependencies
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    with track_time("dist_init"):
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
    lora_peft_config = getattr(config, "state_dict", None)

    with track_time("model_load"):
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            training_args.use_mp,
            get_is_flash_attention_supported(),
            training_args.max_seq_len,
        )
        if lora_peft_config is not None:
            model = get_lora_model_from_base_model(model, lora_peft_config)

        decoder_layer_module = get_submodule_by_pattern(model, r"DecoderLayer$")

    if decoder_layer_module is None:
        track_time("decoder_layer_module_is_none")
        raise ValueError("decoder_layer_module is None.")

    with track_time("model_shard"):
        model = shard_model(
            model,
            decoder_layer_module,
            training_args.use_mp,
            training_args.use_activation_checkpointing,
            training_args.sharding_strategy,
        )

    with track_time("set_activation_checkpointing"):
        if training_args.use_activation_checkpointing:
            hook_activation_checkpointing(model, decoder_layer_module)

    # load dataset
    with track_time("dataset_load"):
        dataset = Dataset(
            config=config.dataset,
            tokenizer=tokenizer,
        )

    # instantiate trainer
    trainer = Trainer(
        config=training_args,
        enable_wandb_logging=config.enable_wandb_logging,
        original_dataset_length=dataset.original_length,
        timer_handle=track_time,
    )

    # load optimizer
    with track_time("optimizer_initialize"):
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
            with track_time("save_final"):
                hf_save_dir = os.path.join(training_args.output_dir, "final-model")
        else:
            with track_time("save_checkpoint"):
                hf_save_dir = os.path.join(
                    training_args.output_dir,
                    "checkpoints",
                    f"epoch_{epoch}",
                    "end-epoch-model",
                )
        with track_time("save_consolidated"):
            save_consolidated_model(trainer.model, hf_save_dir, rank)
            if rank == 0:
                tokenizer.save_pretrained(hf_save_dir)

        dataset.reset_dataloaders()


if __name__ == "__main__":
    args = parse_args()
    config = Config(yaml_path=args.yaml_path)
    setup(config.train_parameters.output_dir)
    main(config, args.model_name)
    cleanup()
