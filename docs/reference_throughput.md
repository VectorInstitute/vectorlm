# Reference Throughput

We've benchmarked VectorLM on the Vaughan cluster for a number of model architectures across a variety of node configurations.
In experiments labelled as LoRA, we set hidden dimension to 8. During the testing, the NVIDIA driver version was 525.105.17, CUDA Runtime 12.1.105, and torch 2.2.2.

For consistency, we use a batch size of 8 and the maximum context length that the pre-trained LLM supports, capped at 65536. Note that especially for smaller models, it might be possible to further increase throughput by switching to a larger batch size.

Entries that read NaN represent combinations where the node configuration does not have enough GPU memory for the training run to complete. An exception is gemma-2b, which currently does not support full-rank FSDP fine-tuning.

All values in the table below represent the median training throughput in tokens/second across all training steps, aggregated across all GPU devices.

|                                      | Meta-Llama-3-8B (2048)   | Meta-Llama-3-8B (4096)   | Meta-Llama-3-8B (8192)   |
|:-------------------------------------|:-------------------------|:-------------------------|:-------------------------|
| (full_rank) NVIDIA A100-SXM4-80GB x1 | 3550.48 (batch: 8)       | 3461.64 (batch: 4)       | 3204.21 (batch: 2)       |
| (full_rank) NVIDIA A100-SXM4-80GB x2 | 6346.00 (batch: 8)       | 6182.59 (batch: 4)       | 5772.91 (batch: 2)       |
| (full_rank) NVIDIA A100-SXM4-80GB x4 | 12688.44 (batch: 8)      | 12249.74 (batch: 4)      | 11463.46 (batch: 2)      |
| (lora) NVIDIA A100-SXM4-80GB x1      | 4079.28 (batch: 8)       | 3682.15 (batch: 4)       | 3528.93 (batch: 2)       |
| (lora) NVIDIA A100-SXM4-80GB x2      | 7182.97 (batch: 8)       | 6955.58 (batch: 4)       | 6452.96 (batch: 2)       |
| (lora) NVIDIA A100-SXM4-80GB x4      | 14299.47 (batch: 8)      | 13834.43 (batch: 4)      | 12769.23 (batch: 2)      |