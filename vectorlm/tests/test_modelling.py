"""
Test model loading, sharding, and forward/backward.
"""

from collections import Counter, defaultdict

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import AdamW
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from vectorlm.utils.model_utils import (
    hook_activation_checkpointing,
    initialize_lora_model_and_tokenizer,
    load_model_and_tokenizer,
    shard_model,
    get_submodule_by_pattern,
)


@pytest.fixture(scope="session")
def setup_and_teardown_torch_process_group():
    # Setup
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
def lora_peft_config():
    """
    Example peft config_dict for LoRA.
    """
    return {
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }


@pytest.fixture(scope="session")
def base_model():
    model, tokenizer = load_model_and_tokenizer("facebook/opt-125m", True, False, 1024)
    return model


@pytest.fixture()
def lora_model(lora_peft_config):
    lora_model, tokenizer = initialize_lora_model_and_tokenizer(
        "facebook/opt-125m", True, False, 1024, lora_peft_config
    )
    return lora_model


@pytest.fixture()
def base_model_sharded(base_model, setup_and_teardown_torch_process_group):
    model_sharded = shard_model(base_model, OPTDecoderLayer, True, True, "FULL_SHARD")
    return model_sharded


@pytest.fixture()
def lora_model_sharded(lora_model, setup_and_teardown_torch_process_group):
    model_sharded = shard_model(lora_model, OPTDecoderLayer, True, True, "FULL_SHARD")
    return FSDP(model_sharded, device_id=torch.cuda.current_device())


@pytest.fixture()
def optimizer_lora_sharded(lora_model_sharded):
    optimizer = AdamW(lora_model_sharded.parameters())
    return optimizer


@pytest.fixture()
def batch():
    batch = {
        "input_ids": torch.zeros((1, 12)),
        "labels": torch.zeros((1, 12)),
        "attention_mask": torch.ones((1, 12)),
    }

    batch = {k: v.type(torch.LongTensor) for k, v in batch.items()}
    batch = {k: v.to(torch.device(0)) for k, v in batch.items()}

    return batch


def test_load_model_and_tokenizer():
    """
    Test load base model and tokenizer.
    """
    model, tokenizer = load_model_and_tokenizer("facebook/opt-125m", True, True, 1024)

    print("type(model): {}".format(type(model)))


def test_load_lora_model_and_tokenizer(lora_peft_config):
    """
    Test load base model and tokenizer.
    """
    lora_model, tokenizer = initialize_lora_model_and_tokenizer(
        "facebook/opt-125m", True, True, 1024, lora_peft_config
    )

    print("type(lora_model): {}".format(type(lora_model)))


def test_match_submodule_by_pattern(base_model):
    """
    Test selecting DecoderLayer class from container.
    """

    submodule = get_submodule_by_pattern(base_model, r"DecoderLayer$")
    assert submodule == OPTDecoderLayer


def test_partition_base_model(base_model, setup_and_teardown_torch_process_group):
    """
    Test partitioning base model (no lora/peft).
    """
    base_model = shard_model(base_model, OPTDecoderLayer, True, True, "FULL_SHARD")

    output_text = []
    for parameter_name, parameter in base_model.named_parameters():
        requires_grad = parameter.requires_grad
        output_text.append("{}\t{}".format(requires_grad, parameter_name))

    with open("output_base.txt", "w") as output_file:
        output_file.write("\n".join(output_text))


def test_get_module_types(lora_model_sharded):
    """
    Output type of each module.
    """
    output_text = []
    print(lora_model_sharded)

    for module_name, module in lora_model_sharded.named_modules():
        output_text.append("{}\t{}".format(module_name, type(module)))

    with open("module_types.txt", "w") as output_file:
        output_file.write("\n".join(output_text))


def test_partition_lora_model(lora_model, setup_and_teardown_torch_process_group):
    """
    Test partitioning lora peft model.
    """
    # # lora.Linear is a submodule of OPTDecoderLayer.
    # for index, module in enumerate(lora_model.modules()):
    #     print(index, module)

    model_sharded = shard_model(
        lora_model, nn.modules.linear.Linear, True, True, "FULL_SHARD"
    )
    model_sharded = FSDP(
        model_sharded, use_orig_params=True, device_id=torch.cuda.current_device()
    )

    requires_grad_counters = defaultdict(Counter)

    output_text = []
    reference_device = None
    for parameter_name, parameter in model_sharded.named_parameters():
        requires_grad = parameter.requires_grad
        requires_grad_counters[requires_grad][parameter_name] += 1
        output_text.append(
            "{}\t{}\t{}".format(requires_grad, parameter.device, parameter_name)
        )

        if reference_device is not None:
            assert parameter.device == reference_device

        reference_device = parameter.device

    with open("output.txt", "w") as output_file:
        output_file.write("\n".join(output_text))


def test_forward_base(base_model_sharded, batch):
    """
    Test forward run of sharded base model.
    """
    base_model_sharded.train()
    output = base_model_sharded(**batch)
    loss = output.loss
    loss.backward()
    print(output)
    print(loss)
    print(loss.shape)


def test_forward_lora(lora_model_sharded, batch):
    """
    Test forward run of sharded lora model.
    """
    lora_model_sharded.train()
    output = lora_model_sharded(**batch)
    loss = output.loss
    print(output)
    print(loss)
    print(loss.shape)


def test_forward_backward_lora(lora_model_sharded, batch):
    """
    Test forward and backward run of sharded lora model.
    """
    lora_model_sharded.train()
    output = lora_model_sharded(**batch)
    loss = output.loss

    loss.backward()

    print(output)
    print(loss)
    print(loss.shape)


def test_train_lora(lora_model_sharded, optimizer_lora_sharded, batch):
    """
    Test N optimization steps on the LoRA sharded model.
    """
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
