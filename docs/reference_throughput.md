# Reference Throughput

We've benchmarked VectorLM on the Vaughan cluster for a number of model architectures across a variety of node configurations.
In experiments labelled as LoRA, we set hidden dimension to 8. Below are version numbers of the testing environment:

```bash
$ pip3 freeze|grep -E "(torch|flash-attn|nvidia)"
flash-attn==2.5.8
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-ml-py==12.550.52
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.3.101
nvidia-nvtx-cu12==12.1.105
torch==2.2.1
```

For each context width and hardware configuration, we experiment with a per-device batch size of 2, 4, and 8. In the table below, we report the batch size that maximizes training throughput. All values in the table represent the median training throughput in tokens/second across all training steps, aggregated across all GPU devices.

|                                      | Meta-Llama-3-8B (2048) | Meta-Llama-3-8B (4096) | Meta-Llama-3-8B (8192) |
| :----------------------------------- | :--------------------- | :--------------------- | :--------------------- |
| (full_rank) NVIDIA A100-SXM4-80GB x1 | 3550.48 (batch: 8)     | 3461.64 (batch: 4)     | 3204.21 (batch: 2)     |
| (full_rank) NVIDIA A100-SXM4-80GB x2 | 6346.00 (batch: 8)     | 6182.59 (batch: 4)     | 5772.91 (batch: 2)     |
| (full_rank) NVIDIA A100-SXM4-80GB x4 | 12688.44 (batch: 8)    | 12249.74 (batch: 4)    | 11463.46 (batch: 2)    |
| (lora) NVIDIA A100-SXM4-80GB x1      | 4079.28 (batch: 8)     | 3682.15 (batch: 4)     | 3528.93 (batch: 2)     |
| (lora) NVIDIA A100-SXM4-80GB x2      | 7182.97 (batch: 8)     | 6955.58 (batch: 4)     | 6452.96 (batch: 2)     |
| (lora) NVIDIA A100-SXM4-80GB x4      | 14299.47 (batch: 8)    | 13834.43 (batch: 4)    | 12769.23 (batch: 2)    |

We provide the tools for evaluating the throughput on different context windows and different hardware/model configuration. Refer to the profiling folder in this repository to get started.