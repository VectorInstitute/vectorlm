from __future__ import annotations

import argparse
import math
import os
import sys
from argparse import Namespace
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist
from torch.optim import AdamW
from tqdm import tqdm
from transformers import set_seed
from vllm import SamplingParams

from vectorlm.dataset import Dataset
from vectorlm.sampling import LoRASamplingEngine
from vectorlm.trainer import Trainer
from vectorlm.utils.data_utils import Config
from vectorlm.utils.misc_utils import cleanup, setup, wandb_setup
from vectorlm.utils.model_utils import (
    get_lora_model_from_base_model,
    get_submodule_by_pattern,
    load_model_and_tokenizer,
    shard_model,
)
from vectorlm.utils.optimizer_utils import get_custom_scheduler
from vectorlm.utils.save_utils import (
    checkpoint_exists,
    get_latest_checkpoint_dir,
    save_consolidated_model,
    save_peft_adapter,
)

if TYPE_CHECKING:
    from vllm import LLM

    from vectorlm.sampling.utils import SynchronizationBarriers


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


def main(
    config: Config,
    local_rank: int | None = None,
    world_size: int | None = None,
    barriers: SynchronizationBarriers | None = None,
    get_vllm_llm: Callable[[], LLM] | None = None,
) -> None:
    """Define the main calling function.

    Args:
    ----
        config: vectorlm config, e.g., loaded from yaml
        local_rank: int, where 0 is root process, one process per accelerator.
        world_size: number of processes.
        barriers: SynchronizationBarriers, required for all processes.
        get_vllm_llm: required only for root process (rank 0).

    """
    if barriers is not None:
        # Wait until vllm engine is fully initialized.
        print(f"rank {local_rank} vllm_init_barrier wait")
        barriers.vllm_init.wait()
        print(f"rank {local_rank} vllm_init_barrier cleared")

    training_args = config.train_parameters
    sampler_config = training_args.get("sampler")

    # set a seed
    set_seed(training_args.seed)

    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()

    # set CUDA related dependencies
    if (local_rank is None) or (world_size is None):
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank  # modify if going beyond one node.
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

    print(f"Rank: {rank}, World size: {world_size}")

    # setup wandb
    if rank == 0 and config.enable_wandb_logging:
        wandb_setup(config, **config.wandb_config)

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        config.model,
        training_args.use_mp,
        training_args.use_flash_attention,
        training_args.max_seq_len,
        local_rank,
        training_args.low_cpu_mem_usage,
    )

    lora_peft_config = getattr(
        config.train_parameters,
        "lora_peft_config",
        None,
    )
    is_peft_adapter_restored = False
    if lora_peft_config is not None:
        peft_adapter_path = None
        # Restore peft adapter from filesystem if available.
        if checkpoint_exists(training_args.output_dir):
            peft_adapter_path = os.path.join(
                training_args.output_dir,
                "checkpoints",
                get_latest_checkpoint_dir(
                    os.path.join(training_args.output_dir, "checkpoints"),
                ),
            )
            is_peft_adapter_restored = True

        model = get_lora_model_from_base_model(
            model,
            lora_peft_config,
            peft_adapter_path,
        )

    decoder_layer_module = get_submodule_by_pattern(model, r"DecoderLayer$")
    assert decoder_layer_module is not None, f"No DecoderLayer found in {model}"
    model = shard_model(
        model,
        decoder_layer_module,
        training_args.use_mp,
        training_args.use_activation_checkpointing,
        training_args.sharding_strategy,
        local_rank,
        training_args.low_cpu_mem_usage,
        is_lora_enabled=(lora_peft_config is not None),
    )

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
        is_peft_adapter_restored,
    )

    if sampler_config is not None:
        # vllm_llm is required only on rank 0.
        vllm_llm = get_vllm_llm() if get_vllm_llm is not None else None
        sampling_engine = LoRASamplingEngine(
            trainer,
            vllm_llm,  # required only for rank 0
            SamplingParams(seed=0, temperature=0),
            barriers,
        )
        trainer.sampling_engine = sampling_engine

    # Checkpoint check. Always call before training.
    # If no checkpoint, it returns 0.
    checkpointed_epoch = trainer.find_checkpoint(training_args.output_dir)

    for epoch in range(checkpointed_epoch, training_args.epochs):
        train_dl_iterator = iter(dataset.train_dataloader)
        for _ in tqdm(
            range(len(dataset.train_dataloader)),
            disable=rank != 0,
            file=sys.__stdout__,
        ):
            batch = next(train_dl_iterator)
            trainer.step(batch, epoch)
            dist.barrier()

        if epoch == training_args.epochs - 1:
            hf_save_dir = os.path.join(training_args.output_dir, "final-model")
        else:
            hf_save_dir = os.path.join(
                training_args.output_dir,
                "checkpoints",
                f"epoch_{epoch}",
                "end-epoch-model",
            )
        # Save base (consolidated) model only when not running peft.
        if lora_peft_config is None:
            save_consolidated_model(trainer.model, hf_save_dir, rank)
        else:
            save_peft_adapter(trainer.model, hf_save_dir)

        dataset.reset_dataloaders()

    sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    config = Config(yaml_path=args.yaml_path)
    setup(config.train_parameters.output_dir)
    main(config)
    cleanup()
