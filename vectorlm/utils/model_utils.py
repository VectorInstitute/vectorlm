from __future__ import annotations

import functools
from typing import Any, Callable

import torch
import torch.distributed as dist
from peft import PeftConfig, PeftModel
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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_peft_model_and_tokenizer(
    path: str,
    use_mp: bool,
    use_fa: bool,
    max_seq_len: int,
    peft_adapter_path: str,
    adapter_name: str = "default",
    is_trainable: bool = False,
    config: PeftConfig | None = None,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load a trained PEFT adapter to the base model and return the PeftModel.

    E.g., a base llama-2-13b-chat-hf w/ adapter named nifty
    ├── adapters_lora
        ├── llama-2-13b-chat-hf+nifty

    Args:
    ----
        path: The path where the model and tokenizer are stored.
        use_mp: Whether to use mixed-precision.
        use_fa: Whether to use Flash Attention 2.
        max_seq_len: The maximum sequence length.
        peft_adapter_path: path to the adapter model, e.g.
            adapters_lora/llama-2-13b-chat-hf+nifty
        adapter_name: e.g. nifty
        is_trainable: train or inference mode
        config: additional configs

    Returns:
    -------
        The PEFT model and tokenizer.

    """
    model, tokenizer = load_model_and_tokenizer(
        path,
        use_mp,
        use_fa,
        max_seq_len,
    )
    peft_model = PeftModel.from_pretrained(
        model,
        peft_adapter_path,
        adapter_name,
        is_trainable,
        config,
    )
    return peft_model, tokenizer


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


def fsdp_config(
    use_mp: bool,
    layer_to_wrap: nn.Module,
    strategy: str,
    local_rank: int,
    low_cpu_mem_usage: bool,
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

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_to_wrap},
    )
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
    layer_to_wrap: nn.Module,
    use_mp: bool,
    use_activation_checkpointing: bool,
    strategy: str,
    local_rank: int,
    low_cpu_mem_usage: bool,
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

    Returns:
    -------
        The sharded module with the requested configurations.

    """
    fsdp_cfg = fsdp_config(
        use_mp, layer_to_wrap, strategy, local_rank, low_cpu_mem_usage,
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
    layer: nn.Module,
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
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn,
    )
