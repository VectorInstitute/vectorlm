import sys
import copy
import torch
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from peft.utils import _get_submodules
from finetune import (
    get_parser, 
    get_accelerate_model,
    get_last_checkpoint
)

# This script will save a full 16-bit version of LoRA adapters merged into the base model
# intended for evaluation frameworks which requires the full model
# adapted from: https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930

def dequantize_model(model, quant_type, dtype, device="cpu"):
    # TODO support 8 bit dequantization
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                weights = dequantize_4bit(module.weight.data, quant_state=module.weight.quant_state, quant_type=quant_type).to(dtype)
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)

                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        model.is_loaded_in_4bit = False        
        return model
    
    
if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--save_dir", type=str, help="Path to save merged model")
    args = parser.parse_args()

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if checkpoint_dir is None:
        print("*** Error: No checkpoint found ***")
        sys.exit(0)

    if not completed_training:
        print("*** Warning: Training was incomplete ***")
    
    # loads peft model matching training forward pass

    model = get_accelerate_model(args, checkpoint_dir)

    # https://github.com/huggingface/peft/pull/851
    # The existing implementation will dequantize, add float lora weights, requantize
    # this is not wholly acceptable, because we lose the 16 bit lora adapter precision upon merge
    # we should maintain float precision: dequantize, add float lora weights, do not requantize
    # however, it seems to work in the PR and we have conflicts with main branch if we change it
    merged_model = model.merge_and_unload() # merge & requant

    # dequant to save pretrained
    merged_model = dequantize_model(merged_model, args.quant_type, 
                             (torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.fp32)))
    print("*** Saving checkpoint in float precision ***")
    merged_model.save_pretrained(args.save_dir, safe_serialization=True)