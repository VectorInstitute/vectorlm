#!/bin/bash
#SBATCH --job-name=llama7b-2
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:4
#SBATCH --output=llama-2-7b.%j.out
#SBATCH --error=llama-2-7b.%j.err
#SBATCH --partition=a100
#SBATCH --qos=your_assigned_qos  # CHANGE
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --time=3-00

export NCCL_IB_DISABLE=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=INFO


torchrun --nnodes=1 --nproc-per-node=4 llama_example.py