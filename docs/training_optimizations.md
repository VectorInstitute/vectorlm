# Training Optimizations

We have worked extensively on implementing different optimizations that helped us minimize both memory footprint and communication volume during training. Throughout the passage below, we employ throughput in `tokens/second` (which encompasses the forward and backward pass) as a measure of what the gain is from a given optimization strategy. The setup we use is 1-2 nodes (exact number specified below) of 4 A100-80GB GPUs with NVLink intranode (within-node) connect and 50 Gbps PCIe v4 internode (between-node) connect. Additionally, these numbers are from finetuning Llama-2 7B on sequences of 1024 context length.

TL;DR: we have gone from a throughput of **~1,500 tokens/s** to **~24,000 tokens/s** on the **same** hardware.

## Naive FSDP

A `FULL_SHARD` (see [docs/config.md](../docs/config.md) for more information on what this is) across 8 A100s (2 nodes) with slow interconnect causes a severe communication bottleneck. The expensive all-gathers and reduce-scatters are happening over 50 Gbps interconnect which greatly slows throughput. With just employing mixed-precision, we had a throughput of ~1,500 tokens/s here.

## Mixed Precision

Mixed-precision using BF16. This (partly) allows us to use Tensor cores. As we will see later, we also need to abide by shape constraints to use Tensor cores. Generally you can use FP16 too, but it requires loss scaling because of loss in precision. BF16 fixes this problem. Using mixed precision involves keeping a model in full precision in memory which gets the updates and a copy of the model in BF16 (or FP16) where computation is done (another advantage is being able to increase mini-batch size because of lowered required memory). See more [here](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). All in all, BF16 allows for much faster computation.

## Tensor Cores

As mentioned earlier, there are certain shapes that we are constrained to in order to use Tensor cores. For A100s, our tensor shapes must be a multiple of 64. This is a reason why we pad our vocabulary embedding matrix to be such a multiple so we can make use of Tensor cores for this large GEMM. See more [here](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#:~:text=Tensor%20Core%20requirements%20by%20cuBLAS,M%2C%20N%2C%20and%20K.&text=Always%20but%20most%20efficient%20with,on%20A100%2C%20multiples%20of%20128.&text=Multiples%20of%208-,Always%20but%20most%20efficient%20with%20multiples%20of,on%20A100%2C%20multiples%20of%2064).

## Activation Checkpointing

The core idea is that during the forward pass, we do not need to hold on to all of the intermediate activations. We can hold on to parts of them, and then during the backward pass we can recompute them on the fly as we need them. While this adds a slight compute overhead, it saves a great memory footprint which allows us to use large micro-batch sizes. With these memory savings, we were able to move training to a single A100 node and avoid the expensive communications from happening over slow interconnect. At this point, along with the previous optimizations, the throughput had risen to ~8,000 tokens/s. See more [here](https://github.com/cybertronai/gradient-checkpointing).

## Flash Attention 2

You can read up a lot on this as there are blog posts that do a far better job than me, but it speeds up the expensive self-attention costs by a lot while also reducing the memory required. This single-handedly upped our throughput from ~8,000 tokens/sec to ~16,000 tokens/sec. Here are some blogs for the original [FA](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) and [FA-2](https://crfm.stanford.edu/2023/07/17/flash2.html).

## Scaling Up - Hybrid FSDP

One of the original problems was that we were having to do expensive all-gathers and reduce-scatters over slow interconnect. One way to avoid it was just to use a single node and avoid the headache altogether, but this hinders our ability to scale up to more data parallel workers. What if we could keep expensive all-gathers and reduce-scatters within a node but scale up and duplicate models across nodes? This is exactly what `HYBRID_SHARD` does. Now, we only have to bare the cost of an all-reduce between nodes which is far cheaper than doing a `FULL_SHARD` across the 2 nodes. This finally brings us to a throughput of ~24,000 tokens/s.