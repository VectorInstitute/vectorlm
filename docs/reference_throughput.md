# Reference Throughput

We've benchmarked VectorLM on the Vaughan cluster for number of model architectures across a variety of node configurations.
In each experiment, we use a batch size of 8 and the maximum context length that the pre-trained LLM supports, capped at 65536.
In experiments labelled as LoRA, we set hidden dimension to 8.

Entries that read NaN represent combinations where the node configuration does not have enough GPU memory for the training run to complete. An exception is gemma-2b, which currently does not support full-rank FSDP fine-tuning.

All values in the table below represent the median training throughput in tokens per second across all training steps, aggregated across all GPU devices.

|                                      | Llama-2-13b-hf | Llama-2-7b-hf | Mistral-7B-v0.1 | Mixtral-8x7B-Instruct-v0.1 | gemma-2b | opt-350m |
| :----------------------------------- | -------------: | ------------: | --------------: | -------------------------: | -------: | -------: |
| (full_rank) NVIDIA A100-SXM4-80GB x1 |        424.726 |       570.818 |         528.747 |                        nan |      nan |  780.045 |
| (full_rank) NVIDIA A100-SXM4-80GB x2 |        660.355 |        919.19 |         794.566 |                    275.459 |      nan |  1227.67 |
| (full_rank) NVIDIA A100-SXM4-80GB x4 |         1309.4 |       1744.39 |         1577.09 |                    817.162 |      nan |  2181.46 |
| (full_rank) NVIDIA A40 x1            |            nan |       47.6435 |         107.503 |                        nan |      nan |  666.881 |
| (full_rank) NVIDIA A40 x2            |            nan |       313.074 |         322.624 |                        nan |      nan |  854.672 |
| (full_rank) NVIDIA A40 x4            |         345.96 |       570.977 |         553.658 |                        nan |      nan |  1765.49 |
| (full_rank) Tesla T4 x1              |            nan |           nan |             nan |                        nan |      nan |   475.51 |
| (full_rank) Tesla T4 x2              |            nan |           nan |             nan |                        nan |      nan |  768.008 |
| (full_rank) Tesla T4 x4              |            nan |           nan |             nan |                        nan |      nan |   1383.6 |
| (full_rank) Tesla T4 x8              |            nan |           nan |             nan |                        nan |      nan |  2414.68 |
| (lora) NVIDIA A100-SXM4-80GB x1      |        560.167 |       646.801 |         525.802 |                        nan |  851.678 |  859.379 |
| (lora) NVIDIA A100-SXM4-80GB x2      |        871.993 |       1157.17 |         1105.68 |                    239.431 |  1724.57 |  1463.82 |
| (lora) NVIDIA A100-SXM4-80GB x4      |        1783.53 |       2091.03 |         2150.06 |                    1309.74 |  2719.24 |  2381.01 |
| (lora) NVIDIA A40 x1                 |        272.931 |       435.386 |         336.507 |                        nan |  983.256 |  652.611 |
| (lora) NVIDIA A40 x2                 |        105.442 |       457.183 |         356.263 |                        nan |  725.723 |  1136.17 |
| (lora) NVIDIA A40 x4                 |         543.22 |       715.416 |         642.642 |                        nan |  1302.62 |  1647.57 |
| (lora) Tesla T4 x1                   |            nan |           nan |             nan |                        nan |  148.272 |  571.471 |
| (lora) Tesla T4 x2                   |            nan |       101.126 |         102.859 |                        nan |  256.534 |  811.159 |
| (lora) Tesla T4 x4                   |            nan |       188.575 |         190.127 |                        nan |  495.755 |  1506.05 |
| (lora) Tesla T4 x8                   |        196.709 |       372.375 |         351.361 |                        nan |   897.81 |  2945.86 |
