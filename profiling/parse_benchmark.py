"""
Parse benchmarking results
to generate metrics overview table.
"""

from collections import defaultdict
import os
import json
import glob

import pandas

benchmark_artifact_folder = "data/benchmark/"

# Load all benchmark result jsonl files
benchmark_jsonl_list = glob.glob("*.jsonl", root_dir=benchmark_artifact_folder)
raw_benchmarks = []
for jsonl_filename in benchmark_jsonl_list:
    jsonl_path = os.path.join(benchmark_artifact_folder, jsonl_filename)
    with open(jsonl_path, "r") as jsonl_file:
        benchmark_content = [
            json.loads(line) for line in jsonl_file.read().splitlines()
        ]
        benchmark_content.append({"name": "_source", "value": jsonl_path})

    raw_benchmarks.append(benchmark_content)

# (model_name, device)
aggregated_output = defaultdict(dict)
for raw_benchmark in raw_benchmarks:
    example_output = {}

    # Need to implement alternative reducing method
    # string: most recent
    # number: summation
    for line in raw_benchmark:
        name = line["name"]
        value = line["value"]
        example_output[name] = value

    model_name = example_output.get("model_name")
    if model_name is None:
        continue

    model_name = model_name.split("/")[-1]
    source_filename = example_output["_source"]

    device_info = example_output["device_info"]
    device_name = device_info["device_name"]
    world_size = device_info["world_size"]
    device_description = "{} x{}".format(device_name, world_size)

    if world_size > 1:
        print(source_filename)

    train_step = example_output.get("train_step")
    if train_step is not None:
        train_throughput = train_step["num_tokens"] / train_step["time_elapsed"]
    else:
        train_throughput = None

    aggregated_output[model_name][device_description] = train_throughput

throughput_table = pandas.DataFrame(aggregated_output).T

print(throughput_table)

with open("data/benchmark/table.md", "w") as table_output_file:
    table_output_file.write(throughput_table.to_markdown())
