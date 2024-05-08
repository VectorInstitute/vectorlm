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
echo "num_gpus: ${num_gpus}; args: $@; nvidia-smi: $(nvidia-smi -L)"
export master_port=$((19000 + SLURM_JOB_ID % 1000))

# select random port
# see pytorch.org/docs/stable/elastic/run.html
torchrun \
--nnodes 1 \
--master-port ${master_port} \
--nproc-per-node ${num_gpus} \
profiling/benchmark.py \
--yaml_path $1 \
--model_name $2 \
--max_length $3 \
--per_device_train_batch_size $4

# clean up benchmarking artifacts as ops have requested
rm -rf /dev/shm/lora-benchmark
