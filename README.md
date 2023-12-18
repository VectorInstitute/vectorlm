# MedGPT
README is a WIP but this is a basic example.
## Installation
We need to install the CUDA 11.8 wheel because torch ships with CUDA 12 (GPUs are currently on CUDA 11).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install .
```

If you would like to use Flash Attention-2, please follow the instructions [here](https://github.com/Dao-AILab/flash-attention).

## Training
Set the respective paths and values in `configs/config.yaml` and run the following on a 4xA100-80G machine:
```bash
export NCCL_IB_DISABLE=1
torchrun --nnodes=1 --nproc-per-node=4 main.py
```