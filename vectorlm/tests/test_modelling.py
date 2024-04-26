"""Test model loading, sharding, and forward/backward."""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from typing import Any, Generator

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim import AdamW
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from vectorlm.utils.model_utils import (
    get_lora_model_from_base_model,
    get_submodule_by_pattern,
    load_model_and_tokenizer,
    shard_model,
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))


@pytest.fixture()
def _setup_and_teardown_torch_process_group() -> Generator[None, None, None]:
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:25567",
        rank=0,
        world_size=1,
    )

    yield

    # Teardown
    dist.destroy_process_group()


@pytest.fixture()
def lora_peft_config() -> dict[str, Any]:
    """Populate example peft config_dict for LoRA."""
    return {
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }


@pytest.fixture()
def base_model() -> torch.nn.Module:
    """Instantiate example non-sharded non-peft transformer model."""
    model, tokenizer = load_model_and_tokenizer(
        "/model-weights/opt-350m",
        True,
        False,
        1024,
        local_rank,
        True,
    )
    return model


@pytest.fixture()
def lora_model(
    base_model: torch.nn.Module,
    lora_peft_config: dict[str, Any],
) -> torch.nn.Module:
    """Obtain LoRA-wrapped base model."""
    return get_lora_model_from_base_model(base_model, lora_peft_config)


@pytest.fixture()
def base_model_sharded(
    base_model: torch.nn.Module,
    _setup_and_teardown_torch_process_group,  # noqa: ANN001
) -> torch.nn.Module:
    """Obtain FSDP-sharded base model."""
    return shard_model(
        base_model,
        OPTDecoderLayer,
        True,
        True,
        "FULL_SHARD",
        local_rank,
        True,
    )


@pytest.fixture()
def lora_model_sharded(
    lora_model: torch.nn.Module,
    _setup_and_teardown_torch_process_group,  # noqa: ANN001
) -> torch.nn.Module:
    """Obtain FSDP-sharded LoRA model."""
    model_sharded = shard_model(
        lora_model,
        OPTDecoderLayer,
        True,
        True,
        "FULL_SHARD",
        local_rank,
        True,
    )
    return FSDP(model_sharded, device_id=torch.cuda.current_device())


@pytest.fixture()
def optimizer_lora_sharded(
    lora_model_sharded: torch.nn.Module,
) -> torch.optim.AdamW:
    """Instantiate optimizer for sharded LoRA model."""
    return AdamW(lora_model_sharded.parameters())


@pytest.fixture()
def batch() -> dict[str, torch.Tensor]:
    """Populate example batch for testing."""
    batch = {
        "input_ids": torch.zeros((1, 12)),
        "labels": torch.zeros((1, 12)),
        "attention_mask": torch.ones((1, 12)),
    }

    batch = {k: v.type(torch.LongTensor) for k, v in batch.items()}
    return {k: v.to(torch.device(0)) for k, v in batch.items()}


def test_load_base_model(base_model: torch.nn.Module) -> None:
    """Ensure no error is encountered when instantiating base model fixture."""
    print(base_model)


def test_match_submodule_by_pattern(base_model: torch.nn.Module) -> None:
    """Test selecting DecoderLayer class from container."""
    submodule = get_submodule_by_pattern(base_model, r"DecoderLayer$")
    assert submodule == OPTDecoderLayer

    submodule = get_submodule_by_pattern(base_model, r"DecoderLayer$")
    assert submodule == OPTDecoderLayer


@pytest.mark.usefixtures("_setup_and_teardown_torch_process_group")
def test_partition_base_model(base_model_sharded: torch.nn.Module) -> None:
    """Test partitioning base model (no lora/peft)."""
    output_text = []
    for parameter_name, parameter in base_model_sharded.named_parameters():
        requires_grad = parameter.requires_grad
        assert requires_grad
        output_text.append(f"{requires_grad}\t{parameter_name}")

    with open("data/output_base.txt", "w") as output_file:
        output_file.write("\n".join(output_text))


def test_get_module_types(lora_model_sharded: torch.nn.Module) -> None:
    """Output type of each module."""
    output_text = []
    print(lora_model_sharded)

    for module_name, module in lora_model_sharded.named_modules():
        output_text.append(f"{module_name}\t{type(module)}")

    with open("data/module_types.txt", "w") as output_file:
        output_file.write("\n".join(output_text))


@pytest.mark.usefixtures("_setup_and_teardown_torch_process_group")
def test_fsdp_lora_model_require_grad(
    lora_model_sharded: torch.nn.Module,
) -> None:
    """Test partitioning lora peft model."""
    requires_grad_counters = defaultdict(Counter)

    output_text = []
    reference_device = None
    for parameter_name, parameter in lora_model_sharded.named_parameters():
        requires_grad = parameter.requires_grad
        requires_grad_counters[requires_grad][parameter_name] += 1
        if re.search("lora_[A|B]", parameter_name) is not None:
            assert requires_grad, parameter_name
        else:
            assert not requires_grad, parameter_name

        output_text.append(
            f"{requires_grad}\t{parameter.device}\t{parameter_name}",
        )

        if reference_device is not None:
            assert parameter.device == reference_device

        reference_device = parameter.device

    with open("data/output.txt", "w") as output_file:
        output_file.write("\n".join(output_text))


def test_forward_base(
    base_model_sharded: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> None:
    """Test forward run of sharded base model."""
    base_model_sharded.train()
    output = base_model_sharded(**batch)
    loss = output.loss
    loss.backward()
    print(output)
    print(loss)
    print(loss.shape)


def test_forward_lora(
    lora_model_sharded: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> None:
    """Test forward run of sharded lora model."""
    lora_model_sharded.train()
    output = lora_model_sharded(**batch)
    loss = output.loss
    print(output)
    print(loss)
    print(loss.shape)


def test_forward_backward_lora(
    lora_model_sharded: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> None:
    """Test forward and backward run of sharded lora model."""
    lora_model_sharded.train()
    output = lora_model_sharded(**batch)
    loss = output.loss

    loss.backward()

    print(output)
    print(loss)
    print(loss.shape)


def test_train_lora(
    lora_model_sharded: torch.nn.Module,
    optimizer_lora_sharded: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> None:
    """Test N optimization steps on the LoRA sharded model."""
    optimizer = optimizer_lora_sharded
    model = lora_model_sharded
    loss_values = []
    for _ in range(7 * 13):
        output = model(**batch)
        loss = output.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.cpu().item())
        print(loss.cpu().item())

    print(loss_values)
    assert loss_values[-1] < loss_values[0]
