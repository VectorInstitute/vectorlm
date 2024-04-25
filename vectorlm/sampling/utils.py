"""Generic utils for the sampling engines."""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, NamedTuple

from vllm import SamplingParams

from .abstract import AbstractSamplingEngine


class SampleOutput(NamedTuple):
    """Represents possible responses to a prompt.

    Params:
        prompt: prompt string.
        options: list of proposed responses to this prompt.
    """

    prompt: str
    options: list[str]
    time_taken: float


def handle_sample(
    sampling_engine: AbstractSamplingEngine,
    prompts: Iterable[str],
    output_path: str | None,
    sampling_params: SamplingParams | None = None,
    extra_data: dict[str, Any] | None = None,
) -> list[SampleOutput]:
    """Sample continuations and optionally save to disk.

    Params:
    ------
        sampling_engine: an instantiation of sampling engine.
        prompts: a list (iterable) of prompts.
        output_path: if provided, append output json lines to this file.
        sampling_params: forwarded to sampling engine.
        extra_data: prepended to each line of output (e.g., current epoch.)

    Returns
    -------
        List of SampleOutput, representing continuations for each prompt.

    """
    _prompts = list(prompts)

    start_time = time.time()
    generation_output = sampling_engine.generate(_prompts, sampling_params)
    time_taken = time.time() - start_time

    # Parse sample engine output and keep only the output strings.
    sample_outputs: list[SampleOutput] = []
    for prompt, options in zip(prompts, generation_output):
        sample_outputs.append(
            SampleOutput(
                prompt,
                [option.text for option in options],
                time_taken,
            ),
        )

    # note: always produce jsonl_output_lines to ensure code coverage.
    extra_data = extra_data if extra_data is not None else {}
    jsonl_output_lines: list[str] = [
        json.dumps({**extra_data, **sample_output._asdict()})
        for sample_output in sample_outputs
    ]
    if output_path is not None:
        with open(output_path, "a") as output_jsonl_file:
            output_jsonl_file.write("\n".join(jsonl_output_lines) + "\n\n")

    return sample_outputs
