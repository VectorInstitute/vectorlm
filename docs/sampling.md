# Efficient Sampling during training

Some training objectives, noteably PPO, require "sampling" from the language model many times during training. The most straightforward approach might be to invoke model.generate on the model from within the training loop. Nevertheless, there have been a number of alternative inference approaches, including vLLM and others, promising over 10x the sampling throughput in terms of tokens generated per second when using a large sampling batch size. If model.generate is taking up too much of the training time, it might be worthwhile looking into these third-party solutions for speeding up the sampling process.

One main challenge of running these third-party solutions, however, is that most of them assume that the weights of the language model are fixed, such that there isn't a straightforward way of updating these weights. Usually, updating the weights requires restarting the sampling engine, which sometimes take minutes. At the same time, the performance of PPO and similar techniques heavily rely on the ability to replace the weights efficiently, or else the training would no longer be on-policy and convergence would take substantially more training steps. To resolve this issue, we implemented techniques to "hot-swap" the model parameters that are used in the sampling process.

Additionally, it is not straightforward to ensure a consistently high GPU utilization when combining sampling with training.
This repository enables you to make the most out of all your GPUs by fitting vLLM and your training loop into the same set of devices. This way, none of the GPUs would sit idle- if a GPU is not running training, it would be busy sampling using vLLM. These slides ([link](https://docs.google.com/presentation/d/1FCa5O8RYYkRRCAAcXhqCvomePo5fEfhjQciSteTEJ30/edit?usp=sharing)) provide an overview of the architecture behind this approach.

## Example- Supervised fine-tuning

We provide a basic example that samples from the language model while fine-tuning using a basic causal language modelling objective. To run the example, uncomment the "sampler" section in your configuration yaml, choose a port for `nccl` coordination, and run the following command (not using torchrun):

```
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=19132
python3 examples/llama_example_mp.py \
--yaml_path configs/config.yaml \
--world_size 2
```

## Bring your own training loop

While the reference implementation is only for supervised fine-tuning, we provide abstractions that make it easier for you to implement your own training loop- be it PPO RLHF, TWIST, or something else. The goal is to abstract away all the synchronization logic, so that a training loop you've built on one GPU could scale to multiple GPUs on the same server with minimal modifications.

To get started, refer to examples/llama_example.py and vectorlm/trainer.py. Usually, the vLLM Engine is accessible only from the rank 0, making synchronization challenging. When invoked through llama_example_mp, the `SamplingEngine` interface in VectorLM enables your training loop to access vLLM.LLM.generate from all ranks, returning the same result across all ranks. Note that because the synchronization barriers require all ranks to reach the synchronization point, you need to invoke `generate` from all ranks.
