"""
Create SLURM jobs running the LoRA benchmark. 
"""

import argparse
import itertools
import subprocess
import time
from os import makedirs
from typing import List

from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--qos", required=False, default="scavenger")
parser.add_argument("--partitions", required=False, default="t4v2,a40")
parser.add_argument("--max_num_jobs", required=False)
launcher_args = parser.parse_args()
qos_selected = launcher_args.qos
max_num_jobs = launcher_args.max_num_jobs
partitions = launcher_args.partitions.split(",")

model_list = [
    "/model-weights/" + model_name
    for model_name in [
        "opt-350m",
        "gemma-2b",
        "Llama-2-7b-hf",
        "Llama-2-13b-hf",
        "Mistral-7B-v0.1",
        "Mixtral-8x7B-Instruct-v0.1",
    ]
]

config_list = [
    "profiling/configs/lora-benchmark.yaml",
    "profiling/configs/benchmark.yaml",
]

slurm_flags_options = {
    "nodes": [1],
    "mem-per-gpu": ["16GB"],
    "ntasks-per-node": [1],
    "cpus-per-gpu": [3],
    "gres": ["gpu:{}".format(n) for n in [1, 2, 4, 8]],
    "partition": partitions,
}

num_repeats = 2
slurm_flags_extra = {"time": "01:00:00", "qos": qos_selected}

slurm_pos_args_options = [["profiling/launch_benchmark.sh"], config_list, model_list]
timestamp = int(time.time())

args_list: List[List[str]] = []
for index, (flag_values, pos_args_option, _) in enumerate(
    itertools.product(
        itertools.product(*(slurm_flags_options.values())),
        itertools.product(*slurm_pos_args_options),
        range(num_repeats)
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
        if value is not None:
            arg = ("--{}".format(key), str(value))
            args.extend(arg)

    args.extend(pos_args_option)
    args_list.append(args)
    print(" ".join(args))

    if (max_num_jobs is not None) and index + 1 >= int(max_num_jobs):
        break

input("\nPress ENTER to launch {} job(s)".format(len(args_list)))

makedirs("data/output", exist_ok=True)
for args in tqdm(args_list, ncols=75):
    subprocess.run(args)
