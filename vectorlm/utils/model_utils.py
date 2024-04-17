from __future__ import annotations

import functools
import re
from typing import Any, Callable

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_half_precision_model(model: nn.Module) -> nn.Module:
    """Cast model to appropriate half-precision format.

    Args:
    ----
        model: nn.Module to cast.

    Returns:
    -------
        nn.Module

    """
    return model.bfloat16()


def get_lora_model_from_base_model(
    base_model: PreTrainedModel,
    peft_config_dict: dict[str, Any],
    peft_adapter_path: str | None = None,
) -> PeftModel:
    """Initialize lora peft configuration from a non-lora model.

    Args:
    ----
        base_model: HuggingFace Transformer model to wrap.
        peft_config_dict: configuration from yaml config file.
        peft_adapter_path: optionally, initialize peft adapters
            using tensors loaded from the filesystem.

    Returns:
    -------
        PeftModel

    """
    task_type_str = peft_config_dict["task_type"]
    task_type = getattr(TaskType, task_type_str)
    lora_config = LoraConfig(**{**peft_config_dict, "task_type": task_type})

    # See github.com/pytorch/pytorch/pull/102212
    base_model.load_state_dict(base_model.state_dict(), assign=True)

    if peft_adapter_path is not None:
        lora_model = PeftModel.from_pretrained(
            base_model,
            peft_adapter_path,
            is_trainable=True,
        )
        print(f"Restored peft_adapter from {peft_adapter_path}.")
    else:
        lora_model = get_peft_model(base_model, lora_config)

    lora_model = get_half_precision_model(lora_model)
    assert isinstance(lora_model, PeftModel)
    lora_model.print_trainable_parameters()
    return lora_model


def load_model_and_tokenizer(
    path: str,
    use_mp: bool,
    use_fa: bool,
    max_seq_len: int,
    local_rank: int,
    low_cpu_mem_usage: bool,
    use_safetensors: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the model and tokenizer.

    Args:
    ----
        path: The path where the model and tokenizer are stored.
        use_mp: Whether to use mixed-precision.
        use_fa: Whether to use Flash Attention 2.
        max_seq_len: The maximum sequence length.
        local_rank: The local rank of the current worker.
        low_cpu_mem_usage: Whether to only load model weights on main rank, and
            then scatter them to the other workers.
        use_safetensors: Whether to use HF safe tensors. Note that this format
            loads significantly faster.
        local_rank: The local rank of the current worker.

    Returns:
    -------
        The model and tokenizer.


    """
    # load model
    model_args = {"use_cache": False, "use_safetensors": use_safetensors}

    if use_mp:
        model_args["torch_dtype"] = torch.bfloat16
    if use_fa:
        if not use_mp:
            msg = "Use FA with bf16 (mixed precision)"
            raise ValueError(msg)
        model_args["attn_implementation"] = "flash_attention_2"

    if not low_cpu_mem_usage or local_rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            **model_args,
        )
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                path,
                **model_args,
            )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = max_seq_len

    # extend embeddings to a multiple so we use Tensor cores
    multiple = 64 if "A100" in torch.cuda.get_device_name() else 8
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=multiple,
    )
    return model, tokenizer


def lora_requires_grad_policy_fn(module: nn.Module) -> bool:
    """Policy that "turns off" FSDP Flat Param for LoRA-enabled layers.

    FSDP requires consistent requires_grad for each flat param.

    Since LoRA requires_grad tensors are embedded within each layer,
    this policy "turns off" FSDP flat param optimization by
    requiring a separate flat param block for each tensor.
    """
    if (
        len(list(module.named_children())) == 0
        and getattr(module, "weight", None) is not None
        and module.weight.requires_grad
    ):
        return True
    return False


def fsdp_config(
    use_mp: bool,
    layer_to_wrap: nn.Module,
    strategy: str,
    local_rank: int,
    low_cpu_mem_usage: bool,
    is_lora_enabled: bool = False,
) -> dict[str, Any]:
    """Get FSDP config.

    Args:
    ----
        use_mp: Whether to use mixed-precision.
        layer_to_wrap: The layer we are wrapping using FSDP.
        strategy: The sharding strategy to use.
        local_rank: The local rank of the current worker.
        low_cpu_mem_usage: Whether to only load model weights on main rank, and
            then scatter them to the other workers.
        is_lora_enabled: Whether to enable LoRA support.

    Returns:
    -------
        A dictionary containing the configurations.

    """

    def _module_init_fn(module: nn.Module) -> Callable:
        """Return the function used for initializing modules on FSDP workers."""
        return module.to_empty(
            device=torch.cuda.current_device(),
            recurse=False,
        )

    strategy_exists = hasattr(ShardingStrategy, strategy)
    if not strategy_exists:
        msg = f"The sharding strategy {strategy} does not exist."
        raise ValueError(msg)

    ret_dict = {}
    if use_mp:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        ret_dict["mixed_precision"] = mp_policy

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_to_wrap},
    )

    if is_lora_enabled:
        # turns off FSDP Flat Param in LoRA layers.
        lambda_requires_grad_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lora_requires_grad_policy_fn,
        )
        auto_wrap_policy = functools.partial(
            _or_policy,
            policies=[lambda_requires_grad_policy, transformer_wrap_policy],
        )
    else:
        auto_wrap_policy = transformer_wrap_policy

    sharding_strategy = getattr(ShardingStrategy, strategy)

    ret_dict["auto_wrap_policy"] = auto_wrap_policy
    ret_dict["sharding_strategy"] = sharding_strategy
    ret_dict["device_id"] = torch.cuda.current_device()
    ret_dict["forward_prefetch"] = True
    if low_cpu_mem_usage:
        ret_dict["param_init_fn"] = _module_init_fn if local_rank != 0 else None
        ret_dict["sync_module_states"] = True
    return ret_dict


def shard_model(
    model: nn.Module,
    layer_to_wrap: type[nn.Module],
    use_mp: bool,
    use_activation_checkpointing: bool,
    strategy: str,
    local_rank: int,
    low_cpu_mem_usage: bool,
    is_lora_enabled: bool = False,
) -> nn.Module:
    """Shard the model to workers using FSDP.

    Args:
    ----
        model: The model to be sharded.
        layer_to_wrap: The layer we are wrapping using FSDP.
        use_mp: Whether to use mixed-precision.
        use_activation_checkpointing: Whether to use activation checkpointing.
        strategy: The sharding strategy to use.
        local_rank: The local rank of the current worker.
        low_cpu_mem_usage: Whether to only load model weights on main rank, and
            then scatter them to the other workers.
        is_lora_enabled: Whether to enable support for LoRA, where only a subset of
            parameter tensors requires_grad. Enabling might significantly reduce
            training throughput, so enable this only when actually using LoRA.

    Returns:
    -------
        The sharded module with the requested configurations.

    """
    fsdp_cfg = fsdp_config(
        use_mp,
        layer_to_wrap,
        strategy,
        local_rank,
        low_cpu_mem_usage,
        is_lora_enabled,
    )
    if dist.get_rank() == 0:
        print(f"FSDP config: {fsdp_cfg}")
    model = FSDP(model, **fsdp_cfg)
    print(
        "Model sharded. Per device model parameters are ",
        f"{sum(p.numel() for p in model.parameters())}",
    )

    if use_activation_checkpointing:
        hook_activation_checkpointing(model, layer_to_wrap)
    return model


def hook_activation_checkpointing(
    model: nn.Module,
    layer: type[nn.Module],
) -> None:
    """Set activation checkpointing.

    Args:
    ----
        model: The model we are using.
        layer: The layer to which we hook activation checkpointing to.

    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    check_fn = lambda submodule: isinstance(submodule, layer)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )


def get_submodule_by_pattern(
    module: nn.Module,
    pattern: str,
) -> type[nn.Module] | None:
    """Return the first module.cls that matches pattern at least partially.

    With reference to get_module_class_from_name from HuggingFace
    accelerate `FullyShardedDataParallelPlugin`.

    Args:
    ----
        module: Layer container
        pattern: regular expression string.

    Returns:
    -------
        nn.Module: matched layer (nn.Module),
        or
        None: if not matched.

    """
    modules_children = list(module.children())
    module_name = module.__class__.__name__
    if re.search(pattern, module_name) is not None:
        return module.__class__

    if len(modules_children) == 0:
        return None

    for child_module in modules_children:
        module_class = get_submodule_by_pattern(child_module, pattern)
        if module_class is not None:
            return module_class

    return None
