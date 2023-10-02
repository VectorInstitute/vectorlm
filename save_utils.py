# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# retrieved from https://github.com/facebookresearch/llama-recipes/blob/2e768b1d1deffd502180e504468edbf6859ec3e1/src/llama_recipes/model_checkpointing/checkpoint_handler.py
# on 29th September 2023. Note that the file has been slightly modified.

from pathlib import Path
from datetime import datetime
import torch
import time
import os
import re

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed.fsdp.api import LocalOptimStateDictConfig

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def extract_model_from_parallel(model, keep_fp32_wrapper: bool = True):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    while isinstance(model, FSDP):
        model = model.module

    return model


def save_metadata(
    out_dir,
    meta_dict,
    tr_step,
    epoch,
):
    save_dir = os.path.join(
        out_dir,
        "checkpoints",
        f"epoch_{epoch}",
        f"checkpoint_{tr_step}",
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(meta_dict, os.path.join(save_dir, "meta_data.pkl"))


def load_metadata(
    in_dir: str,
    rank: int
):
    latest_checkpoint_folder = get_latest_checkpoint_dir(
        os.path.join(in_dir, "checkpoints")
    )
    save_dir = os.path.join(
        in_dir,
        "checkpoints",
        latest_checkpoint_folder,
    )
    if rank == 0:
        print(f"Checkpoint found at {save_dir}")
    save_path = os.path.join(save_dir, "meta_data.pkl")
    meta_dict = torch.load(save_path)
    checkpointed_step = meta_dict["tr_step"]
    checkpointed_epoch = meta_dict["epoch"]
    to_remove = meta_dict["processed_ids"].int().tolist()
    return checkpointed_step, checkpointed_epoch, to_remove


def get_latest_checkpoint_dir(folder_path):
    epoch_pattern = re.compile(r"^epoch_(\d+)$")
    folder_pattern = re.compile(r"^checkpoint_(\d+)$")

    def find_largest(pattern, folder):
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

    epoch_folder = find_largest(epoch_pattern, folder_path)
    folder_path = os.path.join(folder_path, epoch_folder)
    checkpoint_folder = find_largest(folder_pattern, folder_path)
    return os.path.join(epoch_folder, checkpoint_folder)


def checkpoint_exists(output_dir: str):
    if os.path.isdir(os.path.join(output_dir, "checkpoints")):
        return True
    return False


def gather(x: torch.Tensor):
    output_tensors = [x.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, x)
    return torch.cat(output_tensors, dim=0)


def save_model(model, output_dir, rank):
    weights_name = f"model_rank{rank}.bin"
    output_model_file = os.path.join(output_dir, weights_name)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        print(f"Saving model to {output_model_file}")
        state_dict = model.state_dict()
        torch.save(state_dict, output_model_file)
        print(f"Model saved to {output_model_file}")


def save_consolidated_model(model, save_dir, rank):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        unwrapped_model = extract_model_from_parallel(model)
        unwrapped_model.save_pretrained(
            os.path.join(save_dir, "final-model"),
            in_main_process=rank == 0,
            save_function=torch.save,
            max_shard_size="10GB",
            state_dict=unwrapped_model.state_dict(),
        )



def load_model(model, input_dir, rank):
    weights_name = f"model_rank{rank}.bin"
    input_model_file = os.path.join(input_dir, weights_name)
    cfg = LocalStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, cfg):
        print(f"Loading model from {input_model_file}")
        state_dict = torch.load(input_model_file)
        if rank == 0:
            print(f'Loaded model state dict is on {state_dict.device}')
        model.load_state_dict(state_dict)
        print(f"Model loaded from {input_model_file}")


def save_optimizer(optimizer, model, output_dir, rank):
    opt_name = f"optimizer_rank{rank}.bin"
    output_optimizer_file = os.path.join(output_dir, opt_name)
    opt_cfg = LocalOptimStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, optim_state_dict_config=opt_cfg):
        opt_state = FSDP.optim_state_dict(model, optimizer)
        print(f"Saving optimizer state to {output_optimizer_file}")
        torch.save(opt_state, output_optimizer_file)
        print(f"Optimizer state saved to {output_optimizer_file}")


def load_optimizer(optimizer, model, input_dir, rank):
    opt_name = f"optimizer_rank{rank}.bin"
    input_optimizer_file = os.path.join(input_dir, opt_name)
    opt_cfg = LocalOptimStateDictConfig(offload_to_cpu=True)
    model_cfg = LocalStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, model_cfg, opt_cfg):
        print(f"Loading optimizer state from {input_optimizer_file}")
        opt_state = torch.load(input_optimizer_file)
        if rank == 0:
            print(f'Loaded opt state dict is on {opt_state.device}')
        opt_state = FSDP.optim_state_dict_to_load(opt_state, model, optimizer)
        optimizer.load_state_dict(opt_state)
        print(f"Optimizer state loaded from {input_optimizer_file}")
