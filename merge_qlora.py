import sys
from finetune import (
    get_parser, 
    get_accelerate_model,
    get_last_checkpoint
)

# This script will save a full 16-bit version of LoRA adapters merged into the base model
# intended for evaluation frameworks which requires the full model

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
    merged_model = model.merge_and_unload() # dequant & merge
    merged_model.save_pretrained(args.save_dir, safe_serialization=True)