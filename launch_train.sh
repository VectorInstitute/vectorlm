NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# TODO make a config
# /scratch/ssd002/projects/opt_test/llama-30b-hf/llama-30b-hf
# /scratch/ssd002/projects/opt_test/llama-7b-hf
MODEL='/scratch/ssd002/projects/opt_test/llama-7b-hf' # pretrained model
OUTPUT_DIR='/scratch/ssd002/projects/opt_test/clinical_llm/qlora/medgpt-7B-4bit-lr1e-5' # checkpoint saving
SAVE_DIR='/scratch/ssd002/projects/opt_test/clinical_llm/qlora/medgpt-7B-4bit-lr1e-5-merged' # merged checkpoint
LEARNING_RATE='1e-5'
LORA_R=128
BITS=4
BATCH_SIZE=128
BATCH_SIZE_PER_DEVICE=32

# set debug flags
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

# train
torchrun --nproc-per-node $NUM_GPU finetune.py --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR --bits $BITS --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE --lora_r $LORA_R \
    --batch_size_per_device $BATCH_SIZE_PER_DEVICE --save_steps 1000 --eval_steps 50

# merge weights after training
python merge_qlora.py --model_name_or_path $MODEL --output_dir $OUTPUT_DIR --save_dir $SAVE_DIR --lora_r $LORA_R
