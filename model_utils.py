from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch
import functools
import os
import re

class PlateaeuWithWarmup(ReduceLROnPlateau):

    def __init__(self, optimizer, factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, num_warmup_steps=0):
        super().__init__(
            optimizer=optimizer,
            factor=factor,
            threshold=threshold,
            threshold_mode=threshold_mode,
            eps=eps,
            verbose=verbose,
            min_lr=min_lr,
            patience=patience,
            cooldown=cooldown,
        )
        self.num_warmup_steps = num_warmup_steps
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        new_lr = None
        if epoch < self.num_warmup_steps:

            ratio = float(epoch + 1) / float(self.num_warmup_steps)
            new_lr = [ratio * lr for lr in self.base_lrs]
            self._reduce_lr(epoch, new_lr)
        else:

            current = float(metrics)
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def _reduce_lr(self, epoch, new_lr=None):
        if new_lr:
            for i, (lr, param_group) in enumerate(zip(new_lr, self.optimizer.param_groups)):
                param_group['lr'] = lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                    "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
        else:
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        epoch_str = ("%.2f" if isinstance(epoch, float) else
                                    "%.5d") % epoch
                        print('Epoch {}: reducing learning rate'
                            ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))


def load_model_and_tokenizer(cfg):
    model = LlamaForCausalLM.from_pretrained(
        cfg.train.model,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        )
    tokenizer = LlamaTokenizer.from_pretrained(cfg.train.model)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.model_max_length = cfg.train.max_seq_len
    model.resize_token_embeddings(len(tokenizer), 64)
    return model, tokenizer


def fsdp_config():
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )
    sharding_strategy = ShardingStrategy.FULL_SHARD
    ret_dict = {
        "auto_wrap_policy": llama_auto_wrap_policy,
        "mixed_precision": mp_policy,
        "sharding_strategy": sharding_strategy,
        "device_id": torch.cuda.current_device(),
    }
    return ret_dict


def apply_activation_checkpointing(model):
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )