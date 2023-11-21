### MedGPT Project
Authors: Adil Asif, Ziwen Han
Supervisor: John Willes
Collaborators: Ethan Choi, Rahul G. Krishnan


#### Features
- [x] FSDP training
- [ ] DDP + LoRA/QLoRA
- [ ] 
#### Checkpoints
`/checkpoint/opt_test/original/clinical_llm/`


#### Installing bitsandbytes on Vector Cluster
0. Install the requirements
```bash
pip install -r requirements.txt
```

1. Link CUDA and CUDNN libraries. 
We don't need these libraries, the symbolic linking in steps 1 & 2 gives us `nvcc` command the next steps depend on.
Note: Many libraries that require compilation use `nvcc --version` internally, so ensure it matches with your CUDA runtime version.
```bash
ln -s /scratch/ssd001/pkgs/cudnn-11.2-v8.1.1.33 ~/cudnn
ln -s /scratch/ssd001/pkgs/cuda-11.7 ~/cuda
```

2. Modify your `LD_LIBRARY_PATH` and `PATH` in `.bashrc`
```bash
export PATH=~/cuda/bin:~/cudnn/bin:~/.local/bin:~/.local/lib:$PATH
export LD_LIBRARY_PATH=~/cuda/lib64:~/cudnn/lib64:~/x86lib:$LD_LIBRARY_PATH
```
Sanity check: `nvcc --version` should now work.

3. Find your CUDA **runtime** version (not `nvidia-smi` which is the driver)
```bash
pip freeze | grep cuda # nvidia-cuda-runtime-cu11==11.7.99
```

4. Compile bitsandbytes for your CUDA runtime version. Here, I use 117 for the version I detected.
Instructions adapted from: https://github.com/TimDettmers/bitsandbytes
```bash
git clone https://github.com/TimDettmers/bitsandbytes 
CUDA_VERSION=117 make cuda11x
python setup.py install
```

5. Sanity check. It should show success and compiled with cuda.
```bash
python -m bitsandbytes
```

#### Installing FlashAttention 2.0
1. Follow the instructions here:https://github.com/Dao-AILab/flash-attention
2. Note: Ensure nvcc --version displays >= CUDA 11.6 for Driver version.

#### References
- We build on QLoRA: https://github.com/artidoro/qlora

Thanks for open sourcing!