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

import os.path
import shutil
from typing import Generator

import numpy as np
import pytest
import vllm
import vllm.sequence
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest

BASE_MODEL_PATH = "/model-weights/gemma-2b"
LORA_ADAPTER_HF_HUB_REPO = "jacobthebanana/example-gemma-2b-lora-gsm8k"
LORA_ADAPTER_LOCAL_FOLDER = "data/example_lora_adapter"
NUM_TOP_LOGPROBS = 5


@pytest.fixture(scope="session")
def lora_adapter_path() -> str:
    """Download example LoRA adapters from HuggingFace hub.

    Returns
    -------
        Path to the adapters on local filesystem.

    """
    if not os.path.exists(f"{LORA_ADAPTER_HF_HUB_REPO}"):
        snapshot_download(
            LORA_ADAPTER_HF_HUB_REPO,
            local_dir=LORA_ADAPTER_LOCAL_FOLDER,
        )

    return LORA_ADAPTER_LOCAL_FOLDER


@pytest.fixture(scope="session")
def lora_adapter_path_dev_shm(
    lora_adapter_path: str,
) -> Generator[str, None, None]:
    """Create a copy of LoRA adapters on /dev/shm.

    Returns
    -------
        Path to adapters on the /dev/shm filesystem.

    """
    # Specifically require /dev/shm since /tmp might go to an actual disk,
    # incurring overhead and unnecessary SSD wear.
    lora_adapter_dev_shm_path = f"/dev/shm/{LORA_ADAPTER_HF_HUB_REPO}"  # noqa: S108
    os.makedirs(lora_adapter_dev_shm_path, exist_ok=True)
    shutil.copytree(
        lora_adapter_path,
        lora_adapter_dev_shm_path,
        dirs_exist_ok=True,
    )
    print(f"Copy: {lora_adapter_path}, {lora_adapter_dev_shm_path}")

    yield lora_adapter_dev_shm_path

    # Clean up to free memory.
    shutil.rmtree(lora_adapter_dev_shm_path)


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
    return vllm.SamplingParams(
        logprobs=NUM_TOP_LOGPROBS,
        temperature=0.5,
        seed=1,
    )


@pytest.fixture(scope="session")
def example_prompts() -> list[str]:
    """Return example prompts."""
    return [
        "Vector Institute is located in",
        "The answer to life the universe and everything is ",
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


def get_lora_llm_logprobs(
    vllm_model: vllm.LLM,
    example_prompts: list[str],
    vllm_sampling_params: vllm.SamplingParams,
    _lora_adapter_path_fixture_name: str,
    request: pytest.FixtureRequest,
) -> list[list[vllm.sequence.SampleLogprobs]]:
    """Return logprobs for LoRA-adapted LLM."""
    lora_adapter_path = request.getfixturevalue(_lora_adapter_path_fixture_name)
    lora_request = LoRARequest("example_adapter", 1, lora_adapter_path)
    vllm_responses = vllm_model.generate(
        example_prompts,
        vllm_sampling_params,
        lora_request=lora_request,
    )
    return extract_logprobs(vllm_responses)


@pytest.fixture(scope="session")
def lora_llm_logprobs_local_and_dev_shm(
    vllm_model: vllm.LLM,
    example_prompts: list[str],
    vllm_sampling_params: vllm.SamplingParams,
    request: pytest.FixtureRequest,
) -> tuple[list[list[vllm.sequence.SampleLogprobs]], ...]:
    """Return logprobs via LoRA adapter loaded locally and from /dev/shm.

    Returns
    -------
        Two list of lists (options) of vLLM logprobs.
        local_adapter_logprobs, dev_shm_adapter_logprobs

    """
    return tuple(
        get_lora_llm_logprobs(
            vllm_model,
            example_prompts,
            vllm_sampling_params,
            _adapter_path,
            request,
        )
        for _adapter_path in [
            "lora_adapter_path",
            "lora_adapter_path_dev_shm",
        ]
    )


@pytest.fixture(scope="session")
def lora_llm_logprobs_local(
    lora_llm_logprobs_local_and_dev_shm: tuple[
        list[list[vllm.sequence.SampleLogprobs]],
        ...,
    ],
) -> list[list[vllm.sequence.SampleLogprobs]]:
    """Return logprobs from locally-loaded LoRA adapters."""
    return lora_llm_logprobs_local_and_dev_shm[0]


@pytest.fixture(scope="session")
def lora_llm_logprobs_dev_shm(
    lora_llm_logprobs_local_and_dev_shm: tuple[
        list[list[vllm.sequence.SampleLogprobs]],
        ...,
    ],
) -> list[list[vllm.sequence.SampleLogprobs]]:
    """Return logprobs from LoRA adapters loaded via /dev/shm ram-disk."""
    return lora_llm_logprobs_local_and_dev_shm[1]


# Reuse this test case definition across base and LoRA logprobs.
@pytest.mark.parametrize(
    "logprobs_fixture_name",
    [
        "base_llm_logprobs",
        "lora_llm_logprobs_local",
        "lora_llm_logprobs_dev_shm",
    ],
)
def test_logprobs_consistency(
    logprobs_fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Verify consistency of logprobs from base vLLM model.

    Since vLLM seed is fixed, the same prompt should produce
    the same logprobs.
    """
    logprobs: list[list[vllm.sequence.SampleLogprobs]] = (
        request.getfixturevalue(logprobs_fixture_name)
    )

    assert_logprobs_allclose(logprobs[0][0], logprobs[2][0])

    with pytest.raises(AssertionError):
        assert_logprobs_allclose(logprobs[2][0], logprobs[1][0])


def test_compare_ref_logprobs(
    base_llm_logprobs: list[list[vllm.sequence.SampleLogprobs]],
    lora_llm_logprobs_local_and_dev_shm: tuple[
        list[list[vllm.sequence.SampleLogprobs]],
        ...,
    ],
) -> None:
    """Ensure base_llm_logprobs are different from lora_llm_logprobs."""
    # Test both lora_adapter options: disk and ram-disk
    for lora_llm_logprobs in lora_llm_logprobs_local_and_dev_shm:
        for base_llm_seq_logprobs, lora_llm_seq_logprobs in zip(
            base_llm_logprobs,
            lora_llm_logprobs,
        ):
            with pytest.raises(AssertionError):
                assert_logprobs_allclose(
                    base_llm_seq_logprobs[0],
                    lora_llm_seq_logprobs[0],
                )


def test_compare_lora_logprobs(
    lora_llm_logprobs_local_and_dev_shm: tuple[
        list[list[vllm.sequence.SampleLogprobs]],
        ...,
    ],
) -> None:
    """Ensure LoRA logprobs from local and ram-disk adapters match."""
    for logprobs_local_seq, logprobs_dev_shm_seq in zip(
        *lora_llm_logprobs_local_and_dev_shm,
    ):
        # Each of these represents the logprobs options for a sequence output.
        logprobs_local_seq: list[vllm.sequence.SampleLogprobs]
        logprobs_dev_shm_seq: list[vllm.sequence.SampleLogprobs]

        assert_logprobs_allclose(logprobs_local_seq[0], logprobs_dev_shm_seq[0])
        sequence_tokens = "".join(
            [
                str(next(iter(token.values())).decoded_token)
                for token in logprobs_local_seq[0]
            ],
        )
        print(f"\nVerified equivalence: {sequence_tokens}")
