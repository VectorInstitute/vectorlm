"""Generic utils for the sampling engines."""

from __future__ import annotations

import json
import os
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, TypeVar

from vllm import LLM, LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineConfig
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.utils import Counter

from .abstract import AbstractSamplingEngine

if TYPE_CHECKING:
    from threading import Barrier

    from vllm.worker.worker_base import WorkerBase


VECTORLM_WORKER_INIT_RDZV_TIMEOUT = 7


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


class ManagedMultiProcGPUExecutor(MultiprocessingGPUExecutor):
    """MultiProcGPUExecutor, but with VectorLM launched alongside vLLM.

    This class is compatible as an "executor_class" for the vLLM Engine.

    NCCL requires exactly one process for each GPU, so the vLLM and VectorLM
    on each GPU logic need to fit into the same process.

    This class ensures that in each of these one-per-GPU processes,
    VectorLM logic would run in a separate thread alongside the vLLM Worker.
    """

    # only missing parameter in vectorlm_fn is local_rank.
    vectorlm_fn: Callable[[], None]

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
        """Launch vectorlm thread and vLLM worker in the same process.

        For rank 0, this method is invoked "blocking" inside the rank-0 process.

        For rank != 0, this method is supposed to be invoked in a child process
        spawned from the main rank-0 process.
        """
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(local_rank)
        vectorlm_thread = threading.Thread(
            target=self.vectorlm_fn,
            name=f"Rank{local_rank}/vectorlm",
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


class SamplingEngineProvider:
    """Provide VectorLM workers access to the SamplingEngine via a callback.

    The vLLM VectorLM logic needs to be launched alongside the vLLM worker
    This class provides the VectorLM logic

    vLLM engine is initialized only after the initialize_engine call.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        barriers: SynchronizationBarriers,
        sampling_engine_class: AbstractSamplingEngine.__class__,
        vectorlm_main_fn: Callable[
            [Callable[[], AbstractSamplingEngine]],
            None,
        ],
    ) -> None:
        """Instantiate class without initializing wrapped vLLM engine."""
        self.llm_engine: LLMEngine | None = None
        self.llm: LLM | None = None
        self.engine_config = engine_config
        self.barriers = barriers
        self.sampling_engine_class = sampling_engine_class

        # Only missing args is local_rank.
        self.vectorlm_fn: Callable[[], None] = partial(
            vectorlm_main_fn,
            self.get_sampling_engine,
        )

    def initialize_engine(self) -> None:
        """Initialize vLLM engine.

        Invoke this method only from the rank 0 __main__.

        This method blocks until all vectorlm threads (including rank 0)
        have also reached the vllm_init barrier.
        """
        ManagedMultiProcGPUExecutor.vectorlm_fn = self.vectorlm_fn

        self.llm_engine = LLMEngine(
            **self.engine_config.to_dict(),
            executor_class=ManagedMultiProcGPUExecutor,
            log_stats=False,
        )

        self.llm = ManagedLLM(self.llm_engine)
        print(f"Instantiated ManagedLLM: {self.llm}")

        thread_name = threading.current_thread().getName()
        print(f"{thread_name}: vllm_init waiting")

        try:
            self.barriers.vllm_init.wait(VECTORLM_WORKER_INIT_RDZV_TIMEOUT)
        except threading.BrokenBarrierError as e:
            msg = (
                "SamplingEngineProvider requires get_sampling_engine() to be "
                "invoked across all VectorLM ranks (including rank 0) prior "
                "to any Torch NCCL logic. \n"
                "If sampling engine is not required, "
                "please avoid using SamplingEngineProvider, as this provider "
                "would launch vLLM and might hang because of concurrent NCCL "
                "access. Launch the training script via torchrun instead."
            )
            raise RuntimeError(msg) from e

        print(f"{thread_name}: vllm_init cleared")

    def get_sampling_engine(self) -> AbstractSamplingEngine:
        """Instantiate sampling engine.

        Invoke this callback method from the VectorLM thread of each process,
        including rank 0.

        SamplingEngine handles synchronization and prevents concurrent
        NCCL access. Hence, a SamplingEngine instance shall be instantiated
        regardless of the rank of the process.

        This method blocks until the vLLM Engine is fully initialized.
        """
        thread_name = threading.current_thread().getName()
        print(f"{thread_name}: vllm_init wait")
        self.barriers.vllm_init.wait()
        print(f"{thread_name}: vllm_init cleared")

        # vLLM is instantiated and required only for the rank 0 SamplingEngine.
        assert (self.llm is not None) or (int(os.environ["LOCAL_RANK"]) != 0)
        return self.sampling_engine_class(
            self.llm,
            SamplingParams(seed=0, temperature=0),
            self.barriers,
        )

    def join_vectorlm_thread(self) -> None:
        """Join the rank 0 (main process) vectorlm thread.

        Invoke this function only from __main__ (of rank 0) after
        initialize_engine.
        """
        assert self.llm_engine is not None
        model_executor = self.llm_engine.model_executor
        assert isinstance(model_executor, ManagedMultiProcGPUExecutor)
        assert model_executor.driver_worker is not None
        model_executor.driver_worker.vectorlm_thread.join()


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
