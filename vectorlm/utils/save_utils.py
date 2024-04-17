from __future__ import annotations

import os
import re

import peft
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)
from torch.distributed.fsdp import (  # general model non-sharded, non-flattened params
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)

# general model non-sharded, non-flattened params
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def checkpoint_exists(output_dir: str) -> bool:
    """Check if a checkpoint exists.

    Args:
    ----
        output_dir: The main saving directory.

    Returns:
    -------
        Returns whether a checkpoint exists.

    """
    if os.path.isdir(os.path.join(output_dir, "checkpoints")):
        return True
    return False


def save_metadata(
    out_dir: str,
    meta_dict: dict[str, int | torch.Tensor],
) -> None:
    """Save training metadata.

    Args:
    ----
        out_dir: The directory to save to.
        meta_dict: The dictionary containing the meta data.

    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save(meta_dict, os.path.join(out_dir, "meta_data.pkl"))


def load_metadata(
    in_dir: str,
) -> tuple[int, int, list[int]]:
    """Load training metadata.

    Args:
    ----
        in_dir: The directory where the meta data is saved.

    Returns:
    -------
        A tuple containing the checkpointed step, epoch, and the processed
            training dataset ids.

    """
    save_path = os.path.join(in_dir, "meta_data.pkl")
    meta_dict = torch.load(save_path)
    checkpointed_step = meta_dict["tr_step"]
    checkpointed_epoch = meta_dict["epoch"]
    to_remove = meta_dict["processed_ids"].int().tolist()
    return checkpointed_step, checkpointed_epoch, to_remove


def get_latest_checkpoint_dir(folder_path: str) -> str:
    """Find the latest checkpoint directory using regex.

    Args:
    ----
        folder_path: The path to where checkpoints are saved.

    Returns:
    -------
        The subpath (i.e. two levels) of the latest checkpoint's directory.

    """
    epoch_pattern = re.compile(r"^epoch_(\d+)$")
    folder_pattern = re.compile(r"^checkpoint_(\d+)$")

    def _find_largest(pattern: re.Pattern, folder: str) -> str:
        max_integer = -1
        max_folder_name = None

        for folder_name in os.listdir(folder):
            match = pattern.match(folder_name)
            if match:
                current_integer = int(match.group(1))
                if current_integer > max_integer:
                    max_integer = current_integer
                    max_folder_name = folder_name
        return max_folder_name

    epoch_folder = _find_largest(epoch_pattern, folder_path)
    folder_path = os.path.join(folder_path, epoch_folder)
    checkpoint_folder = _find_largest(folder_pattern, folder_path)
    return os.path.join(epoch_folder, checkpoint_folder)


def save_consolidated_model(
    model: nn.Module,
    save_dir: str,
    rank: int,
) -> None:
    """Save the sharded model's parameters consolidated under a single file.

    Args:
    ----
        model: The sharded model.
        save_dir: The checkpointing directory.
        rank: The worker's rank.

    """
    os.makedirs(save_dir, exist_ok=True)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    save_path = os.path.join(save_dir, "model.bin")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
        if rank == 0:
            torch.save(state_dict, save_path)


def get_peft_adapter_tensor_dict(
    model: peft.peft_model.PeftModel,
) -> dict[str, torch.Tensor] | None:
    """Return LoRA PEFT Adapter tensor state dict on rank 0.

    Returns None for all other ranks.
    """
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        if dist.get_rank() == 0:
            return peft.utils.save_and_load.get_peft_model_state_dict(model)

        return None


def save_peft_adapter(
    model: peft.peft_model.PeftModel,
    output_path: str,
) -> None:
    """Save peft adapter to filesystem in a FSDP environment."""
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        if dist.get_rank() == 0:
            model.save_pretrained(output_path)


def save_model_and_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
    output_dir: str,
    rank: int,
) -> None:
    """Save model and optimizer states.

    Args:
    ----
        optimizer: The sharded optimizer.
        model: The sharded model.
        output_dir: The checkpointing directory.
        rank: The worker's rank.

    """
    os.makedirs(output_dir, exist_ok=True)
    opt_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        optim_state_dict_config=opt_cfg,
    ):
        state_dict = model.state_dict()
        opt_state = FSDP.sharded_optim_state_dict(model, optimizer)
        full_state = {"model_state": state_dict, "optim_state": opt_state}
    writer = FileSystemWriter(output_dir, single_file_per_rank=True)
    if _should_save(rank, model.sharding_strategy):
        if rank == 0:
            print(f"Saving states to {output_dir}")
        save(
            state_dict=full_state,
            storage_writer=writer,
            process_group=model.process_group,
            planner=DefaultSavePlanner(),
        )
        if rank == 0:
            print(f"States saved to {output_dir}")


def load_model_and_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
    input_dir: str,
) -> None:
    """Load the model and optimizer states.

    Args:
    ----
        optimizer: The sharded optimizer.
        model: The sharded model.
        input_dir: The checkpointing directory.

    """
    if dist.get_rank() == 0:
        print(f"Loading states from {input_dir}")
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model.state_dict()
        checkpoint = {"model_state": model_state_dict}
        load(
            state_dict=checkpoint,
            storage_reader=FileSystemReader(input_dir),
            planner=DefaultLoadPlanner(),
        )
        model.load_state_dict(checkpoint["model_state"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=model.state_dict(),
            optimizer_key="optim_state",
            storage_reader=FileSystemReader(input_dir),
        )
        flattened_osd = FSDP.optim_state_dict_to_load(
            model,
            optimizer,
            optim_state["optim_state"],
        )
        optimizer.load_state_dict(flattened_osd)
    if dist.get_rank() == 0:
        print(f"States loaded from {input_dir}")


def save_scheduler(
    scheduler: LRScheduler,
    output_dir: str,
    rank: int,
) -> None:
    """Save scheduler states.

    Args:
    ----
        scheduler: The LR scheduler.
        output_dir: The checkpointing directory.
        rank: The worker's rank.

    """
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        sched_name = "scheduler.bin"
        output_scheduler_file = os.path.join(output_dir, sched_name)
        print(f"Saving scheduler state to {output_scheduler_file}")
        state_dict = scheduler.state_dict()
        torch.save(state_dict, output_scheduler_file)
        print(f"Scheduler state saved to {output_scheduler_file}")


def load_scheduler(
    scheduler: LRScheduler,
    input_dir: str,
    rank: int,
) -> None:
    """Load scheduler states.

    Args:
    ----
        scheduler: The LR scheduler.
        input_dir: The checkpointing directory.
        rank: The worker's rank.

    """
    sched_name = "scheduler.bin"
    input_scheduler_file = os.path.join(input_dir, sched_name)
    if rank == 0:
        print(f"Loading scheduler state from {input_scheduler_file}")
    state_dict = torch.load(input_scheduler_file)
    scheduler.load_state_dict(state_dict)
    if rank == 0:
        print(f"Scheduler state loaded from {input_scheduler_file}")


def _should_save(rank: int, strategy: ShardingStrategy) -> bool:
    """Whether we should save on this rank.

    In HSDP, we only save on one of the shard_group
    (i.e. non-replicated ranks).

    Args:
    ----
        rank: The global rank.
        strategy: The sharding strategy for FSDP.

    Returns:
    -------
        Whether we should save on this rank.

    """
    if strategy == ShardingStrategy.HYBRID_SHARD:
        local_rank = int(os.environ["LOCAL_RANK"])
        return local_rank == rank
    return True
