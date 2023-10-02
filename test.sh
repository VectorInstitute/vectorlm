NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun --nproc-per-node $NUM_GPU finetune.py --model_name_or_path '/scratch/ssd002/projects/opt_test/llama-7b-hf' &