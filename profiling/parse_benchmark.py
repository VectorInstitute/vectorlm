"""Parse benchmarking results to generate metrics overview table."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar, Union

import numpy as np
import pandas as pd

Numbers = Union[int, float]
NumericalTypes = Union[Numbers, np.ndarray]
V = TypeVar("V")
Aggregator = TypeVar("Aggregator")
Numerical = TypeVar("Numerical", bound=NumericalTypes)


# Skip first N train steps (warmup, profiling, etc.) in throughput eval.
NUM_SKIPPED_STEPS = 80


@dataclass
class RunningAverage:
    """Abstraction for tracking numbers required to compute averages.

    Params:
        running_count: number of observations added

    """

    running_count: int = 0
    running_sum: NumericalTypes | None = None

    def add(self, observation: NumericalTypes) -> None:
        """Add observation to accumulator.

        Params
        ------
            observation: must be numerical and of same type
            (number or np.ndarray) as running_sum.

        """
        self.running_count += 1
        if self.running_sum is None:
            self.running_sum = observation
        else:
            self.running_sum += observation

    def get_average(self) -> NumericalTypes | None:
        """Obtain average of this accumulator.

        Returns
        -------
        NumericalTypes
            same type (number or np.ndarray) as self.running_sum.

        """
        if (self.running_count == 0) or (self.running_sum is None):
            return None

        return self.running_sum / self.running_count


def _reduce_metric(
    new_value: NumericalTypes | str | dict,
    previous_value: NumericalTypes | str | dict | list,
) -> NumericalTypes | str | dict | list:
    """Recursively reduce values.

    Params
    ------
        new_value: value to aggregate
        previous_value: aggregator

    Returns
    -------
        Same type as previous value.

    """
    if isinstance(new_value, (float, int)):
        if not isinstance(previous_value, list):
            previous_value = [previous_value]

        return [*previous_value, new_value]

    if isinstance(new_value, dict) and isinstance(previous_value, dict):
        for k in previous_value:
            if k in new_value:
                previous_value[k] = _reduce_metric(
                    new_value[k],
                    previous_value[k],
                )

        return previous_value

    return new_value


def get_quantiles(values: list[Numbers]) -> np.ndarray:
    """Given a list of numerical values, return (min, 25%, 50%, 75%, 95%, max).

    Params
    ------
        values: list of numerical values, must be non-empty.

    Returns
    -------
        np.ndarray.

    """
    percentiles = [0.25, 0.5, 0.75, 0.95]

    if len(values) == 0:
        return [np.nan] * (1 + len(percentiles) + 1)

    output_list = [
        np.min(values),
        *[np.percentile(values, q) for q in percentiles],
        np.max(values),
    ]

    return np.asarray(output_list)


parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="data/benchmark/")
args = parser.parse_args()
benchmark_artifact_folder = args.folder

# Load all benchmark result jsonl files
benchmark_jsonl_list = glob.glob("*.jsonl", root_dir=benchmark_artifact_folder)
raw_benchmarks = []
for jsonl_filename in benchmark_jsonl_list:
    jsonl_path = os.path.join(benchmark_artifact_folder, jsonl_filename)
    with open(jsonl_path) as jsonl_file:
        benchmark_content = [
            json.loads(line) for line in jsonl_file.read().splitlines()
        ]
        benchmark_content.append({"name": "_source", "value": jsonl_path})

    raw_benchmarks.append(benchmark_content)

# Set of tuples the form (model_name, device)
benchmarked_combinations: set[tuple[str, str]] = set()
# Map (model, device) pair to dict mapping (batch_size, seq_len) to aggregator.
aggregated_output: dict[tuple[str, str], dict[str, RunningAverage]] = (
    defaultdict(
        lambda: defaultdict(lambda: RunningAverage()),
    )
)
profiler_tables = defaultdict(dict)

# Aggregate benchmark files to obtain average values
# for each model-device combination.
for raw_benchmark in raw_benchmarks:
    benchmark_output = {}

    # If an entry (e.g., train_step) is logged multiple times
    # in the benchmark output, aggregate these values.
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
    context_window = benchmark_output.get("max_length")
    if model_name is None:
        continue

    model_name = model_name.split("/")[-1]
    model_name = f"{model_name} ({context_window})"
    source_filename = benchmark_output["_source"]

    peft_method = benchmark_output.get("peft_method")
    if peft_method is None:
        continue

    device_info = benchmark_output["device_info"]
    device_name = device_info["device_name"]
    if isinstance(device_info["world_size"], list):
        world_size = device_info["world_size"][0]
    else:
        world_size = device_info["world_size"]
    device_description = f"({peft_method}) {device_name} x{world_size}"

    # Training throughput can be noisy. Report median throughput,
    # and discard instances with only one training step logged.
    train_step = benchmark_output.get("train_step")
    if train_step is not None:
        num_tokens = np.asarray(train_step["num_tokens"])
        time_elapsed = np.asarray(train_step["time_elapsed"])
        if num_tokens.flatten().shape[0] > 1:
            train_throughput = get_quantiles(
                (num_tokens / time_elapsed)[NUM_SKIPPED_STEPS:],
            )
            aggregated_output[(model_name, device_description)][
                "batch: " + str(benchmark_output.get("training_batch_size"))
            ].add(
                train_throughput[2],
            )

    # torch profiler output in tabular format
    benchmarked_combinations.add((model_name, device_description))
    profiler_table_str = benchmark_output.get("profiler_table")
    if profiler_table_str is not None:
        profiler_tables[model_name][device_description] = profiler_table_str


aggregated_output_nested = defaultdict(dict)
for combination in benchmarked_combinations:
    model_name, device_description = combination
    # there might be more than one run for each batch size option
    # average median throughput over all runs for each option.
    # report batch size that achieves optimal (avg) throughput.
    throughput: list[tuple[NumericalTypes, str]] = [
        (average, batch_size)
        for (average, batch_size) in (
            (aggregation.get_average(), batch_size)
            for batch_size, aggregation in aggregated_output[
                combination
            ].items()
        )
        if average is not None
    ]
    if len(throughput) == 0:
        continue

    optimal_throughput, optimal_batch_size = sorted(throughput, reverse=True)[0]
    aggregated_output_nested[model_name][device_description] = (
        f"{optimal_throughput:.2f} ({optimal_batch_size})"
    )


throughput_table = (
    pd.DataFrame(aggregated_output_nested)
    .sort_index(axis="columns")
    .sort_index(axis="index")
)
print(throughput_table)

table_output_lines: list[str] = []
markdown_output_path = os.path.join(benchmark_artifact_folder, "table.md")
with open(markdown_output_path, "w") as table_output_file:
    table_output_lines.append(throughput_table.to_markdown())
    table_output_file.write("\n".join(table_output_lines))

print(f"\nWriting summary to {markdown_output_path}")
