# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# retrieved from https://github.com/facebookresearch/llama-recipes/blob/2e768b1d1deffd502180e504468edbf6859ec3e1/src/llama_recipes/model_checkpointing/checkpoint_handler.py
# on 29th September 2023. Note that the file has been slightly modified.

import torch
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

def save_consolidated_model(model, save_dir, rank):
    os.makedirs(save_dir, exist_ok=True)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    save_path = os.path.join(save_dir, 'model.bin')
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
        if rank == 0:
            torch.save(state_dict, save_path)