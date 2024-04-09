"""
Parse benchmarking results
to generate metrics overview table.
"""

import argparse
from collections import defaultdict
import os
import json
import glob
from typing import List, Dict, TypeVar, Tuple, List, Union
import numpy as np
from dataclasses import dataclass

import pandas

Numbers = Union[int, float]
V = TypeVar("V")
Aggregator = TypeVar("Aggregator")

@dataclass
class RunningAverage:
    running_count: int = 0
    running_sum: Union[Numbers, np.ndarray] = None

    def add(self, observation):
        self.running_count += 1
        if self.running_sum is None:
            self.running_sum = observation
        else:
            self.running_sum += observation

    def get_average(self) -> Union[Numbers, List[Numbers]]:
        if self.running_count == 0:
            return None
            
        average = self.running_sum / self.running_count
        if hasattr(average, "tolist"):
            return average.tolist()

        return average


def _reduce_metric(new_value, previous_value):
    """
    Recursively reduce values.
    """
    if isinstance(new_value, (float, int)):
        if not isinstance(previous_value, list):
            previous_value = [previous_value]

        return [*previous_value, new_value]

    elif isinstance(new_value, dict) and isinstance(previous_value, dict):
        for k in previous_value.keys():
            if k in new_value.keys():
                previous_value[k] = _reduce_metric(new_value[k], previous_value[k])

        return previous_value
    else:
        return new_value


def get_quantiles(values: List[Numbers]) -> np.ndarray:
    """
    Given a list of numeraical values,
    return (min, 25%, 50%, 75%, and max).

    Params:
        values: list of numerical values, must be non-empty.

    Returns:
        np.ndarray.
    """
    output_list = [
        np.min(values),
        *[np.percentile(values, q) for q in [0.25, 0.5, 0.75]],
        np.max(values),
    ]

    return np.asarray(output_list)


parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="data/benchmark/")
args = parser.parse_args()
benchmark_artifact_folder = args.folder

# Load all benchmark result jsonl files
benchmark_jsonl_list = glob.glob("*.jsonl", root_dir=benchmark_artifact_folder)
print(benchmark_jsonl_list)
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
benchmarked_combinations = set()
aggregated_output: Dict[Tuple[str, str], RunningAverage] = defaultdict(lambda: RunningAverage())
profiler_tables = defaultdict(dict)
for raw_benchmark in raw_benchmarks:
    benchmark_output = {}

    for line in raw_benchmark:
        name = line["name"]
        value = line["value"]
        previous_value = benchmark_output.get(name)
        if previous_value is not None:
            new_value = _reduce_metric(value, previous_value)
        else:
            new_value = value

        benchmark_output[name] = new_value

    model_name = benchmark_output.get("model_name")
    if model_name is None:
        continue

    model_name = model_name.split("/")[-1]
    source_filename = benchmark_output["_source"]

    peft_method = benchmark_output.get("peft_method")
    if peft_method is None:
        continue
    
    device_info = benchmark_output["device_info"]
    device_name = device_info["device_name"]
    world_size = device_info["world_size"]
    device_description = "{} x{} {}".format(device_name, world_size, peft_method)

    train_step = benchmark_output.get("train_step")
    if train_step is not None:
        num_tokens = np.asarray(train_step["num_tokens"])
        time_elapsed = np.asarray(train_step["time_elapsed"])
        train_throughput = get_quantiles(
            world_size * num_tokens / time_elapsed
        )
        aggregated_output[(model_name, device_description)].add(train_throughput)

    benchmarked_combinations.add((model_name, device_description))
    profiler_table_str = benchmark_output.get("profiler_table")
    if profiler_table_str is not None:
        profiler_tables[model_name][device_description] = profiler_table_str


aggregated_output_nested = defaultdict(dict)
for combination in benchmarked_combinations:
    model_name, device_description = combination
    throughput = aggregated_output[combination].get_average()
    aggregated_output_nested[model_name][device_description] = throughput
    

throughput_table = pandas.DataFrame(aggregated_output_nested)
throughput_table.sort_index(axis="columns", inplace=True)
throughput_table.sort_index(axis="index", inplace=True)
print(throughput_table)

table_output_lines: List[str] = []
markdown_output_path = os.path.join(benchmark_artifact_folder, "table.md")
with open(markdown_output_path, "w") as table_output_file:
    table_output_lines.append(throughput_table.to_markdown())

    model_names = sorted(list(profiler_tables.keys()))
    for model_name in model_names:
        table_output_lines.append("\n## {}".format(model_name))
        profiler_table_dict = profiler_tables[model_name]
        device_descriptions = sorted(list(profiler_table_dict.keys()))

        for device_description in device_descriptions:
            profiler_table_str = profiler_table_dict[device_description]
            table_output_lines.append("### {}".format(device_description))
            table_output_lines.append("```\n{}\n```".format(profiler_table_str))

    table_output_file.write("\n".join(table_output_lines))

print("\nWriting summary to {}".format(markdown_output_path))
