#!/bin/bash
#SBATCH --job-name=llama7b-2-lora
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:1
#SBATCH --output=llama-2-7b-lora.%j.out
#SBATCH --error=llama-2-7b-lora.%j.err
#SBATCH --partition=a100
#SBATCH --qos=your_assigned_qos  # CHANGE
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --time=3-00

export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN

# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Uncomment these flags for debugging communication
# export TORCH_CPP_LOG_LEVEL=INFO
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=0

torchrun --nnodes=1 --nproc-per-node=1 example_lora.py --yaml_path configs/config-lora.yaml
