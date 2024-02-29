from __future__ import annotations

import os
import re

import torch
from peft.peft_model import PeftModel
from torch import nn
from torch.distributed.fsdp import (
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import LocalOptimStateDictConfig
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
    """Find the latest checkpoing directory using regex.

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


def save_model(model: nn.Module, output_dir: str, rank: int) -> None:
    """Save the sharded model's weights.

    Args:
    ----
        model: The sharded model.
        output_dir: The checkpointing directory.
        rank: The worker's rank.
    """
    os.makedirs(output_dir, exist_ok=True)
    weights_name = f"model_rank{rank}.bin"
    output_model_file = os.path.join(output_dir, weights_name)

    if isinstance(model, PeftModel):
        model.save_pretrained(output_model_file)
        return

    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        print(f"Saving model to {output_model_file}")
        state_dict = model.state_dict()
        torch.save(state_dict, output_model_file)
        print(f"Model saved to {output_model_file}")


def load_model(model: nn.Module, input_dir: str, rank: int) -> None:
    """Load the sharded model's weights.

    Args:
    ----
        model: The sharded model.
        input_dir: The checkpointing directory.
        rank: The worker's rank.
    """
    weights_name = f"model_rank{rank}.bin"
    input_model_file = os.path.join(input_dir, weights_name)
    cfg = LocalStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, cfg):
        print(f"Loading model from {input_model_file}")
        state_dict = torch.load(input_model_file)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {input_model_file}")


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
    if isinstance(model, PeftModel):
        model.save_pretrained(save_dir)
        return

    os.makedirs(save_dir, exist_ok=True)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    save_path = os.path.join(save_dir, "model.bin")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
        if rank == 0:
            torch.save(state_dict, save_path)


def save_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
    output_dir: str,
    rank: int,
) -> None:
    """Save the optimizer states.

    Args:
    ----
        optimizer: The sharded optimizer.
        model: The sharded model.
        output_dir: The checkpointing directory.
        rank: The worker's rank.
    """
    opt_name = f"optimizer_rank{rank}.bin"
    output_optimizer_file = os.path.join(output_dir, opt_name)
    opt_cfg = LocalOptimStateDictConfig(offload_to_cpu=True)

    try:
        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
            optim_state_dict_config=opt_cfg,
        ):
            opt_state = FSDP.optim_state_dict(model, optimizer)

            print(f"Saving optimizer state to {output_optimizer_file}")
            torch.save(opt_state, output_optimizer_file)
            print(f"Optimizer state saved to {output_optimizer_file}")

    except AttributeError:
        # One GPU only. Optimizer isn't sharded.
        opt_state = optimizer.state_dict()
        print("Optimizer state is retrieved as non-sharded")

        print(f"Saving optimizer state to {output_optimizer_file}")
        torch.save(opt_state, output_optimizer_file)
        print(f"Optimizer state saved to {output_optimizer_file}")


def load_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
    input_dir: str,
    rank: int,
) -> None:
    """Load the optimizer states.

    Args:
    ----
        optimizer: The sharded optimizer.
        model: The sharded model.
        input_dir: The checkpointing directory.
        rank: The worker's rank.
    """
    opt_name = f"optimizer_rank{rank}.bin"
    input_optimizer_file = os.path.join(input_dir, opt_name)
    opt_cfg = LocalOptimStateDictConfig(offload_to_cpu=True)
    model_cfg = LocalStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.LOCAL_STATE_DICT,
        model_cfg,
        opt_cfg,
    ):
        print(f"Loading optimizer state from {input_optimizer_file}")
        opt_state = torch.load(input_optimizer_file)
        opt_state = FSDP.optim_state_dict_to_load(opt_state, model, optimizer)
        optimizer.load_state_dict(opt_state)
        print(f"Optimizer state loaded from {input_optimizer_file}")


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
    sched_name = f"scheduler_rank{rank}.bin"
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
    sched_name = f"scheduler_rank{rank}.bin"
    input_scheduler_file = os.path.join(input_dir, sched_name)
    print(f"Loading scheduler state from {input_scheduler_file}")
    state_dict = torch.load(input_scheduler_file)
    scheduler.load_state_dict(state_dict)
    print(f"Scheduler state loaded from {input_scheduler_file}")
