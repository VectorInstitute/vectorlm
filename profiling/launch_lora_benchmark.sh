#!/bin/bash

export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN

# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Uncomment these flags for debugging communication
# export TORCH_CPP_LOG_LEVEL=INFO
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=0

source ~/vectorlm/env/bin/activate
export PYTHONPATH=$PYTHONPATH:`pwd`

nvidia-smi
export num_gpus=`nvidia-smi -L | wc -l`
echo num_gpus: ${num_gpus}

torchrun \
--nnodes=1 \
--nproc-per-node=${num_gpus} profiling/benchmark.py \
--yaml_path profiling/configs/lora-benchmark.yaml \
--model_name $1

# # clean up benchmarking artifacts as ops have requested
rm -rf /dev/shm/lora-benchmark