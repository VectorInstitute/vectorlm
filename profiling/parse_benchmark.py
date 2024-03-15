"""
Parse benchmarking results
to generate metrics overview table.
"""

import argparse
from collections import defaultdict
import os
import json
import glob
from typing import List

import pandas


parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="data/benchmark/")
args = parser.parse_args()
benchmark_artifact_folder = args.folder

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
profiler_tables = defaultdict(dict)
for raw_benchmark in raw_benchmarks:
    benchmark_output = {}

    # Need to implement alternative reducing method
    # string: most recent
    # number: summation
    for line in raw_benchmark:
        name = line["name"]
        value = line["value"]
        benchmark_output[name] = value

    model_name = benchmark_output.get("model_name")
    if model_name is None:
        continue

    model_name = model_name.split("/")[-1]
    source_filename = benchmark_output["_source"]

    device_info = benchmark_output["device_info"]
    device_name = device_info["device_name"]
    world_size = device_info["world_size"]
    device_description = "{} x{}".format(device_name, world_size)

    train_step = benchmark_output.get("train_step")
    if train_step is not None:
        train_throughput = (
            world_size * train_step["num_tokens"] / train_step["time_elapsed"]
        )
    else:
        train_throughput = None

    aggregated_output[model_name][device_description] = train_throughput
    profiler_table_str = benchmark_output.get("profiler_table")
    if profiler_table_str is not None:
        profiler_tables[model_name][device_description] = profiler_table_str

throughput_table = pandas.DataFrame(aggregated_output).T
throughput_table.sort_index(axis="columns", inplace=True)
throughput_table.sort_index(axis="index", inplace=True)
print(throughput_table)

table_output_lines: List[str] = []
with open(
    os.path.join(benchmark_artifact_folder, "table.md"), "w"
) as table_output_file:
    table_output_lines.append(throughput_table.to_markdown())

    for model_name, profiler_table_dict in profiler_tables.items():
        table_output_lines.append("\n## {}".format(model_name))
        for device_description, profiler_table_str in profiler_table_dict.items():
            table_output_lines.append("### {}".format(device_description))
            table_output_lines.append("```\n{}\n```".format(profiler_table_str))

    table_output_file.write("\n".join(table_output_lines))
