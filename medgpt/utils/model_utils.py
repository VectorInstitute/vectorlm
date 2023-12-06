from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import torch
from typing import Tuple, Any

def load_model_and_tokenizer(
    path: str,
    use_mp: bool,
    use_fa: bool,
    max_seq_len: int,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """"""

    # load model
    model_args = {}

    if use_mp:
        model_args["torch_dtype"] = torch.bfloat16
    if use_fa:
        if not use_mp:
            raise ValueError("Use FA with bf16 (mixed precision)")
        model_args["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(
        path,
        **model_args,
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    if isinstance(model, LlamaForCausalLM):
        # Llama requires us to add the pad token
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.model_max_length = max_seq_len

    # extend embeddings to a multiple so we use Tensor cores
    if "A100" in torch.cuda.get_device_name():
        multiple = 64
    else:
        multiple = 8
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=multiple,
    )
    return model, tokenizer


def fsdp_config(
    use_mp: bool,
    layer_to_wrap: torch.nn.Module
) -> dict[str, Any]:
    """"""
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
    sharding_strategy = ShardingStrategy.FULL_SHARD

    ret_dict["auto_wrap_policy"] = auto_wrap_policy
    ret_dict["sharding_strategy"] = sharding_strategy
    ret_dict["device_id"] = torch.cuda.current_device()
    return ret_dict


def hook_activation_checkpointing(
    model: torch.nn.Module,
    layer: torch.nn.Module,
) -> None:
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    check_fn = lambda submodule: isinstance(submodule, layer)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )