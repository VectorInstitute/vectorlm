#!/bin/bash
#SBATCH --job-name=llama7b-2-multinode
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:4
#SBATCH --output=llama-2-7b.%j.out
#SBATCH --error=llama-2-7b.%j.err
#SBATCH --partition=a100
#SBATCH --qos=a100_adilasif
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --time=3-00

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Uncomment these flags for debugging communication
# export TORCH_CPP_LOG_LEVEL=INFO

srun -p $SLURM_JOB_PARTITION \
    -c $SLURM_CPUS_ON_NODE \
    -N $SLURM_JOB_NUM_NODES \
    --mem=0 \
    --gres=gpu:$SLURM_JOB_PARTITION:$SLURM_GPUS_ON_NODE \
    bash -c 'torchrun \
    --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv-id $RDVZ_ID \
    --rdzv-backend c10d \
    llama_example.py --yaml_path ../configs/config.yaml'