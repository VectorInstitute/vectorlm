"""Generic utils for the sampling engines."""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, TypeVar

from vllm import LLM
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.utils import Counter

from .abstract import AbstractSamplingEngine

if TYPE_CHECKING:
    from threading import Barrier

    from vllm import LLMEngine, SamplingParams
    from vllm.worker.worker_base import WorkerBase


class SampleOutput(NamedTuple):
    """Represents possible responses to a prompt.

    Params:
        prompt: prompt string.
        options: list of proposed responses to this prompt.
    """

    prompt: str
    options: list[str]
    time_taken: float


class SynchronizationBarriers(NamedTuple):
    """Barriers for limiting GPU access concurrency.

    Params:
        vllm_init: Barrier to Ensures that vLLM engine is fully initialized
            before running any vectorlm logic.

        before_generation: Ensure all processes have reached this statement,
            or vectorlm in some processes might still be accessing the
            accelerator when rank 0 invokes vLLM.

        after_generation: Detain all processes until after rank 0 is sure that
            there are no outstanding vLLM jobs.
    """

    vllm_init: Barrier
    before_generation: Barrier
    after_generation: Barrier


class ManagedMultiProcGPUExecutor(MultiprocessingGPUExecutor):
    """MultiProcGPUExecutor, but with VectorLM launched alongside vLLM."""

    # only missing parameter in vectorlm_fn is local_rank.
    vectorlm_fn: Callable[[int], None]

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        """Copy class variable vectorlm_fn into this instance.

        Doing so ensures that spawned sub-processes also have access
        to vectorlm_fn, which might not otherwise be accessible as a class
        variable.
        """
        self.vectorlm_fn = ManagedMultiProcGPUExecutor.vectorlm_fn
        super().__init__(*args, **kwargs)

    def _create_worker(
        self,
        local_rank: int = 0,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> WorkerBase:
        """Instantiate worker and launch vectorlm thread.

        For rank 0, this method is invoked "blocking" inside the rank-0 process.

        For rank != 0, this method is supposed to be invoked in a child process
        spawned from the main rank-0 process.
        """
        vectorlm_thread = threading.Thread(
            target=self.vectorlm_fn,
            kwargs={"local_rank": local_rank},
        )
        vectorlm_thread.start()

        worker = super()._create_worker(*args, **kwargs, local_rank=local_rank)
        assert worker is not None
        worker.vectorlm_thread = vectorlm_thread

        return worker


class ManagedLLM(LLM):
    """vllm.entrypoints.LLM but using an externally-initialized LLM Engine."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        """Instantiate LLM instance using externally-initialized LLM Engine."""
        self.llm_engine = llm_engine
        self.request_counter = Counter()


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
            Recommended: specify output_path only on rank 0.
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
    for prompt, request_output in zip(prompts, generation_output):
        sample_outputs.append(
            SampleOutput(
                prompt,
                [option.text for option in request_output.outputs],
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


Fn = TypeVar("Fn", bound=Callable[..., Any])


def multiprocess_wrap(fn: Fn, barriers: SynchronizationBarriers) -> Fn:
    """Apply barrier to function and broadcast output.

    This wrapper function tries to preserve the type signature
    of the wrapped function for the IDE. Tested for Pylance.

    While fn would be invoked only on rank 0, the wrapped function
    should be invoked in the vectorlm thread at all ranks, so that
    the barriers would block these threads from accessing GPU while
    the fn is running.

    Each rank would receive the same value as output.

    Params:
    -------
        fn: Function to wrap. Output needs to be compatible with pickle.
        barriers: SynchronizationBarriers, only the before_generation and
            after_generation barriers are required..

    Returns
    -------
        same output as Fn, but broadcasted to all ranks
        (i.e., same value at all ranks)

    """

    def _wrapped_fn(*args, **kwargs) -> ...:  # noqa: ANN002,ANN003
        barriers.after_generation.wait()

        import torch.distributed  # type: ignore[reportMissingImports]

        rank = torch.distributed.get_rank()

        # placeholder for output value,
        # populate on rank 0 and then broadcast.
        # torch requires placeholder element in object list.
        output = [None]
        if rank == 0:
            output = [fn(*args, **kwargs)]

        # fn might access torch.dist, which might conflict with
        # broadcast_object_list. Hence, keep all ranks witing until fn returns
        # on rank 0.
        barriers.before_generation.wait()

        torch.distributed.broadcast_object_list(output)
        return output[0]

    return _wrapped_fn  # type: ignore[reportReturnType]
