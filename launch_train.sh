NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun --nproc-per-node $NUM_GPU finetune.py --model_name_or_path '/scratch/ssd002/projects/opt_test/llama-7b-hf' --output_dir '/scratch/ssd002/projects/opt_test/clinical_llm/qlora/medgpt-7B-4bit-lr1e-5' --learning_rate '1e-5' --lora_r 128 --batch_size_per_device 32 --save_steps 1000 --eval_steps 50 --bits 4 & 
# /scratch/ssd002/projects/opt_test/llama-30b-hf/llama-30b-hf
# /scratch/ssd002/projects/opt_test/llama-7b-hf