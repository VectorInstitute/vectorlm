"""
Create SLURM jobs running the LoRA benchmark. 
"""

from typing import List
import itertools
import subprocess
import time

model_list = [
    "/model-weights/" + model_name
    for model_name in [
        "opt-350m",
        "Llama-2-7b-hf",
        "Llama-2-13b-hf",
        "Mistral-7B-v0.1",
        "t5-xl-lm-adapt",
    ]
]

slurm_flags_options = {
    "nodes": [1],
    "mem": [0],
    "ntasks-per-node": [1],
    "cpus-per-gpu": [6],
    "gres": ["gpu:{}".format(n + 1) for n in range(1)],
    "partition": ["t4v2", "a40", "a100"],
}

slurm_flags_extra = {"time": "00:30:00", "qos": "scavenger"}

slurm_pos_args_options = [["examples/launch_lora_benchmark.sh"], model_list]
timestamp = int(time.time())

for index, (flag_values, pos_args_option) in enumerate(
    zip(
        itertools.product(*(slurm_flags_options.values())),
        itertools.product(*slurm_pos_args_options),
    )
):
    args: List[str] = ["sbatch"]

    extra_flags = {
        **slurm_flags_extra,
        "output": "data/output/{}.{}.out".format(timestamp, index),
        "error": "data/output/{}.{}.out".format(timestamp, index),
        "job-name": "bench-{}-{}".format(timestamp, index),
    }

    keys = list(slurm_flags_options.keys()) + list(extra_flags.keys())
    values = list(flag_values) + list(extra_flags.values())
    for key, value in zip(keys, values):
        arg = ("--{}".format(key), str(value))
        args.extend(arg)

    args.extend(pos_args_option)

    print(" ".join(args))
