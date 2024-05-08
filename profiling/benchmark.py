from __future__ import annotations

# Renamed from examples/llama_example.py
import argparse
import contextlib
import json
import math
import os
import sys
import time
from argparse import Namespace
from typing import Any, Generator

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.profiler import ProfilerActivity
from tqdm import tqdm
from transformers import PreTrainedTokenizer, set_seed

from vectorlm.dataset import Dataset
from vectorlm.trainer import Trainer
from vectorlm.utils.data_utils import Config
from vectorlm.utils.misc_utils import cleanup, setup, wandb_setup
from vectorlm.utils.model_utils import (
    get_lora_model_from_base_model,
    get_submodule_by_pattern,
    hook_activation_checkpointing,
    load_model_and_tokenizer,
    shard_model,
)
from vectorlm.utils.optimizer_utils import get_custom_scheduler
from vectorlm.utils.save_utils import save_consolidated_model

JSONSerializable = str | dict[str, Any] | list[str] | float | None
_MIN_FLASH_ATTENTION_CUDA_CAPABILITY = 8

# Cap value ot tokenizer.model_max_length to this value,
# unless overridden when instantiating the benchmarking dataset.
_MAX_SEQ_LENGTH = 65536


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
    parser.add_argument(
        "--num_train_examples",
        default=10000,
    )
    parser.add_argument(
        "--num_eval_examples",
        default=1000,
    )
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    return parser.parse_args()


# unix timestamp
launch_time = time.time()
output_path = f"data/benchmark/{launch_time}.jsonl"
profiler_output_path = f"data/trace/{launch_time}.json"


def write_metrics(
    metric_name: str,
    value: JSONSerializable = None,
) -> None:
    """Write metric and time elapsed to output file.

    This function writes to disk only if process rank is 0.

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
def track_time(
    task_name: str,
    extra_info: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """Context manager for recording time spent in a code block.

    Params
    ------
        task_name: str
        extra_info: Optional, JSON-serializable dictionary
            to include in log output.

    """
    start_time = time.time()
    try:
        yield
    finally:
        time_elapsed = time.time() - start_time
        metric_value = {"time_elapsed": time_elapsed}
        if extra_info is not None:
            metric_value = {**metric_value, **extra_info}

        write_metrics(task_name, metric_value)


def get_device_info() -> dict[str, str | int]:
    """Get CUDA info as a dict.

    Returns
    -------
        Dict including device_name and world size

    """
    return {
        "device_name": torch.cuda.get_device_name(),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
    }


def get_is_flash_attention_supported() -> bool:
    """Determine whether flash attention is available."""
    version_major, _ = torch.cuda.get_device_capability()
    return version_major >= _MIN_FLASH_ATTENTION_CUDA_CAPABILITY


def get_slurm_env() -> dict[str, str]:
    """Return a dictionary of all env var starting with "SLURM_"."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLURM_")
    }


def parse_profiler_output(
    profiler_output: torch.autograd.profiler.profile,
) -> dict[str, dict[str, str | float | int]]:
    """Parse profiler_output to obtain dictionary of metrics.

    Returns
    -------
        Dictionary mapping event name to dictionary of metrics.

    """
    key_average_event_list = profiler_output.key_averages()
    output: dict[str, dict[str, str | float | int]] = {}
    for evt in key_average_event_list:
        trace_name = getattr(evt, "trace_name", None)
        if trace_name is None:
            continue
        output[evt.trace_name] = {
            "start": evt.time_range.start,
            "elapsed": evt.time_range.elapsed_us(),
            "args": (
                evt.thread
                if not evt.is_remote
                else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "'
            ),
        }

    return output


def _handle_profiler_trace(
    profiler_output: torch.autograd.profiler.profile,
) -> None:
    """Log torch profile to disk.

    This function is to be invoked as a callback for on_track_ready.

    Args:
    ----
        profiler_output: from Torch profiler.

    """
    print(profiler_output)
    key_average_event_list = profiler_output.key_averages()
    write_metrics("profiler_table", key_average_event_list.table())
    parsed_output = parse_profiler_output(profiler_output)
    write_metrics("profiler_output", parsed_output)

    if bool(os.environ.get("PROFILER_EXPORT_TRACE")):
        profiler_output.export_chrome_trace(profiler_output_path)


class BenchmarkingDataset(Dataset):
    """In-memory dataset for benchmarking."""

    def __init__(
        self,
        config: Config,
        num_train_examples: int,
        num_eval_examples: int,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
    ) -> None:
        """Initialize in-memory dataset for benchmarking.

        Refer to vectorlm.dataset for details regarding config
        and tokenizer.

        Params:
        ------
            config: dataset config. Forwarded to vectorlm.dataset.Dataset.
            num_train_examples: length of train split.
            num_eval_examples: length of eval split.
            tokenizer: HuggingFace tokenizer.
            max_length: optional. If not specified,
                fall back to tokenizer.model_max_length.
        """
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples

        if (max_length is not None) and (max_length > 0):
            self.max_length = max_length
        else:
            self.max_length = min(tokenizer.model_max_length, _MAX_SEQ_LENGTH)

        super().__init__(config, tokenizer)

    def load_datasets(self) -> None:
        """Load datasets into memory."""
        self.train_ds, self.eval_ds = (
            [
                {
                    "id": row_id,
                    "input_ids": torch.zeros(self.max_length),
                    "labels": torch.zeros(self.max_length),
                    "attention_mask": torch.ones(self.max_length),
                }
                for row_id in range(length)
            ]
            for length in (self.num_train_examples, self.num_eval_examples)
        )

        self.original_length = math.ceil(len(self.train_ds) / self.train_bs)


if __name__ == "__main__":
    args = parse_args()
    config = Config(yaml_path=args.yaml_path)

    os.makedirs("data/benchmark", exist_ok=True)
    os.makedirs("data/trace", exist_ok=True)

    setup(config.train_parameters.output_dir)

    training_args = config.train_parameters

    # set a seed
    set_seed(training_args.seed)

    # set CUDA related dependencies
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if args.per_device_train_batch_size is not None:
        config.dataset.train_bs = args.per_device_train_batch_size

    write_metrics("training_batch_size", config.dataset.train_bs)
    write_metrics(
        "training_batch_size_global",
        config.dataset.train_bs * world_size,
    )

    print(f"Writing metrics to {output_path}")
    write_metrics("model_name", args.model_name)
    write_metrics("config", {**config.__dict__})
    write_metrics("device_info", get_device_info())
    write_metrics("slurm_info", get_slurm_env())

    profiler_schedule = torch.profiler.schedule(
        skip_first=10,
        wait=5,
        warmup=1,
        active=3,
        repeat=2,
    )

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
    lora_peft_config = config.train_parameters.get("lora_peft_config")
    is_lora_enabled = lora_peft_config is not None

    with track_time("model_load"):
        model, tokenizer = load_model_and_tokenizer(
            args.model_name,
            training_args.use_mp,
            get_is_flash_attention_supported(),
            args.max_length,
            local_rank,
            training_args.low_cpu_mem_usage,
        )
        if is_lora_enabled:
            print("Enabling LoRA Wrapper.")
            write_metrics("peft_method", "lora")
            model = get_lora_model_from_base_model(model, lora_peft_config)

        else:
            write_metrics("peft_method", "full_rank")

        model = model.bfloat16()
        decoder_layer_module = get_submodule_by_pattern(model, r"DecoderLayer$")

    if decoder_layer_module is None:
        msg = "decoder_layer_module is None."
        raise ValueError(msg)

    with track_time("model_shard"):
        model = shard_model(
            model,
            decoder_layer_module,
            training_args.use_mp,
            training_args.use_activation_checkpointing,
            training_args.sharding_strategy,
            local_rank,
            training_args.low_cpu_mem_usage,
            is_lora_enabled=is_lora_enabled,
        )

    with track_time("set_activation_checkpointing"):
        if training_args.use_activation_checkpointing:
            hook_activation_checkpointing(model, decoder_layer_module)

    # load dataset
    with track_time("dataset_load"):
        dataset = BenchmarkingDataset(
            config=config.dataset,
            num_train_examples=args.num_train_examples,
            num_eval_examples=args.num_eval_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        print(
            f"Sequence length: {dataset.max_length};"
            f"Batch Size (per device): {config.dataset.train_bs}",
        )
        write_metrics("max_length", dataset.max_length)

    # instantiate trainer
    trainer = Trainer(
        config=training_args,
        enable_wandb_logging=config.enable_wandb_logging,
        original_dataset_length=dataset.original_length,
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

    trainer.dataset.setup_dataloaders()
    checkpointed_epoch = 0

    # See pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        on_trace_ready=_handle_profiler_trace,
    ) as profile_handle:
        for epoch in range(checkpointed_epoch, training_args.epochs):
            trainer.model.train()
            train_dl_iterator = iter(dataset.train_dataloader)
            for _ in tqdm(
                range(len(dataset.train_dataloader)),
                disable=rank != 0,
                file=sys.__stdout__,
            ):
                batch = next(train_dl_iterator)
                num_tokens = len(batch["input_ids"].flatten())

                with track_time("train_step", {"num_tokens": num_tokens}):
                    trainer.step(batch, epoch)

                profile_handle.step()
                write_metrics(
                    "torch.cuda.utilization",
                    torch.cuda.utilization(),
                )

            if epoch == training_args.epochs - 1:
                with track_time("save_final"):
                    hf_save_dir = os.path.join(
                        training_args.output_dir,
                        "final-model",
                    )
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

    cleanup()
