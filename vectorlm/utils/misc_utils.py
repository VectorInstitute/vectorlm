from __future__ import annotations

import os
from typing import Any

import torch.distributed as dist
import wandb

from vectorlm.utils.data_utils import Config


def setup(final_model_dir: str) -> None:
    """Initialize the process group and create directories."""
    os.makedirs(
        os.path.join(final_model_dir, "final-model"), exist_ok=True,
    )
    dist.init_process_group("nccl")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()


def wandb_setup(config: Config, **kwargs: dict[str, Any]) -> None:
    """Initialize wandb and logging metrics."""
    full_config_dict = {}
    for k, v in config.__dict__.items():
        full_config_dict[k] = v
    wandb.init(
        config=full_config_dict,
        **kwargs,
    )
    wandb.define_metric("millions_of_tokens")
    wandb.define_metric("train/*", step_metric="millions_of_tokens")
    wandb.define_metric("test/*", step_metric="millions_of_tokens")
