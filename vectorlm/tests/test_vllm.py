"""vLLM Integration tests.

LLM for scaffolding:
- gemma-2b (vLLM supports LoRA for Gemma but not OPT)

Scaffolding and fixtures:
- vLLM
    - Spin up vLLM Engine via Python API
    - LoRA request via peft LoRA adapter from
        - regular disk.
        - folder saved in /dev/shm.
- References top-k log probabilities via vLLM
    - Base model
    - LoRA model loaded from /dev/shm
    - LoRA model loaded from disk
"""

from __future__ import annotations

import numpy as np
import pytest
import vllm
import vllm.sequence
from vllm.lora.request import LoRARequest

BASE_MODEL_PATH = "/model-weights/gemma-2b"
LORA_ADAPTER_PATH = "data/example-adapters/gemma-2b-gsm8k"
NUM_TOP_LOGPROBS = 5


@pytest.fixture(scope="session")
def vllm_model() -> vllm.LLM:
    """Spin up vLLM base model."""
    return vllm.LLM(
        BASE_MODEL_PATH,
        gpu_memory_utilization=0.3,
        enable_lora=True,
    )


@pytest.fixture(scope="session")
def vllm_sampling_params() -> vllm.SamplingParams:
    """Return example vLLM sampling parameters for consistency across tests."""
    return vllm.SamplingParams(logprobs=NUM_TOP_LOGPROBS, temperature=0, seed=1)


@pytest.fixture(scope="session")
def example_prompts() -> list[str]:
    """Return example prompts."""
    return [
        "Vector Institute is located in",
        "The answer to life the universe and everything is of course",
        "Vector Institute is located in",
    ]


def extract_logprobs(
    vllm_responses: list[vllm.RequestOutput],
) -> list[list[vllm.sequence.SampleLogprobs]]:
    """Extract logprobs from vllm response.

    Additionally, ensures none of these output is None.

    Params
    ------
        vllm_responses: output from LLM.generate()

    Returns
    -------
        Nested list, one list of logprobs instance for each prompt.

    """
    logprobs_responses: list[list[vllm.sequence.SampleLogprobs]] = []
    for response in vllm_responses:
        for output in response.outputs:
            logprobs_options: list[vllm.sequence.SampleLogprobs] = []
            logprobs = output.logprobs
            assert logprobs is not None
            logprobs_options.append(logprobs)

        logprobs_responses.append(logprobs_options)

    return logprobs_responses


def assert_logprobs_allclose(
    logprobs_a: vllm.sequence.SampleLogprobs,
    logprobs_b: vllm.sequence.SampleLogprobs,
) -> None:
    """Ensure that logprobs_a are all close with logprobs_b."""
    assert len(logprobs_a) == len(logprobs_b)
    for token_logprobs_a, token_logprobs_b in zip(logprobs_a, logprobs_b):
        assert token_logprobs_a.keys() == token_logprobs_b.keys()
        token_logprobs_a_array = np.asarray(
            [token_logprobs_a[k].logprob for k in token_logprobs_a],
        )
        token_logprobs_b_array = np.asarray(
            [token_logprobs_b[k].logprob for k in token_logprobs_a],
        )
        assert np.allclose(
            np.asarray(token_logprobs_a_array),
            np.asarray(token_logprobs_b_array),
        )


@pytest.fixture(scope="session")
def base_llm_logprobs(
    vllm_model: vllm.LLM,
    example_prompts: list[str],
    vllm_sampling_params: vllm.SamplingParams,
) -> list[list[vllm.sequence.SampleLogprobs]]:
    """Return logprobs for base LLM (no LoRA adapter)."""
    vllm_responses = vllm_model.generate(example_prompts, vllm_sampling_params)
    return extract_logprobs(vllm_responses)


@pytest.fixture(scope="session")
def lora_request() -> LoRARequest:
    """Return LoRARequest for vLLM LoRA requests."""
    return LoRARequest("example_adapter", 1, LORA_ADAPTER_PATH)


@pytest.fixture(scope="session")
def lora_llm_logprobs(
    vllm_model: vllm.LLM,
    example_prompts: list[str],
    vllm_sampling_params: vllm.SamplingParams,
    lora_request: LoRARequest,
) -> list[list[vllm.sequence.SampleLogprobs]]:
    """Return logprobs for LoRA-adapted LLM."""
    vllm_responses = vllm_model.generate(
        example_prompts,
        vllm_sampling_params,
        lora_request=lora_request,
    )
    return extract_logprobs(vllm_responses)


@pytest.mark.parametrize(
    "logprobs_fixture_name",
    ["base_llm_logprobs", "lora_llm_logprobs"],
)
def test_get_logprobs(
    logprobs_fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test obtaining logprobs from base vLLM model."""
    output_logprobs: list[list[vllm.sequence.SampleLogprobs]] = (
        request.getfixturevalue(logprobs_fixture_name)
    )
    assert_logprobs_allclose(output_logprobs[0][0], output_logprobs[2][0])

    with pytest.raises(AssertionError):
        assert_logprobs_allclose(
            output_logprobs[2][0],
            output_logprobs[1][0],
        )


def test_compare_ref_logprobs(
    base_llm_logprobs: list[list[vllm.sequence.SampleLogprobs]],
    lora_llm_logprobs: list[list[vllm.sequence.SampleLogprobs]],
) -> None:
    """Ensure base_llm_logprobs are different from lora_llm_logprobs."""
    for base_llm_seq_logprobs, lora_llm_seq_logprobs in zip(
        base_llm_logprobs,
        lora_llm_logprobs,
    ):
        with pytest.raises(AssertionError):
            assert_logprobs_allclose(
                base_llm_seq_logprobs[0],
                lora_llm_seq_logprobs[0],
            )
