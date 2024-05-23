"""Generic utils for the sampling engines."""

from __future__ import annotations

import json
import os
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, TypeVar

from vllm import LLM
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.executor.multiproc_worker_utils import (
    ResultHandler,
    WorkerMonitor,
)
from vllm.utils import (
    Counter,
    get_distributed_init_method,
    get_ip,
    get_open_port,
    get_vllm_instance_id,
)
from vllm.worker.worker import Worker

from .abstract import AbstractSamplingEngine
from .vllm_worker_utils import ManagedProcessWorkerWrapper

if TYPE_CHECKING:
    from threading import Barrier

    from vllm import LLMEngine, SamplingParams


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
    """MultiProcGPUExecutor, but with VectorLM supplied."""

    vectorlm_fn: Callable[[int], None]

    def _init_executor(self) -> None:
        """Launch vectorlm logic in workers.

        Supply barriers and pickle-compatible vectorlm main fn to
        workers via vLLM multiprocessing messaging mechanisms.
        """
        assert (
            not self.speculative_config
        ), "Speculative decoding not yet supported for MultiProcGPU backend."

        # Create the parallel GPU workers.
        world_size = self.parallel_config.tensor_parallel_size

        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, range(world_size))
            )

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        from torch.cuda import device_count

        assert (
            world_size <= device_count()
        ), "please set tensor_parallel_size to less than max local gpu count"

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port()
        )

        if world_size == 1:
            self.workers = []
        else:
            result_handler = ResultHandler()
            self.workers = [
                ManagedProcessWorkerWrapper(
                    result_handler,
                    partial(
                        self._create_worker,
                        rank=rank,
                        local_rank=rank,
                        distributed_init_method=distributed_init_method,
                    ),
                    partial(self.vectorlm_fn, rank),
                )
                for rank in range(1, world_size)
            ]

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self.driver_worker = self._create_worker(
            distributed_init_method=distributed_init_method,
        )
        self.rank_0_vectorlm_thread = threading.Thread(
            target=partial(self.vectorlm_fn, 0),
        )
        self.rank_0_vectorlm_thread.start()

        self._run_workers("init_device")
        self._run_workers(
            "load_model",
            max_concurrent_workers=self.parallel_config.max_parallel_loading_workers,
        )


class VectorLMWorker(Worker):
    """Worker for running VectorLM logic alongside vLLM worker.

    Use this instance for the rank 0 (root) process.

    Note that nccl requires that only one process may have access
    to each GPU. Each LocalWorkerVllm is a multiprocessing.Process.
    Vectorlm logic would be launched as a thread within each of these
    proceses.

    Spawn no more than one such instance for each GPU.

    Attributes
    ----------
        vectorlm_thread: threading.Thread.

    """

    barriers: SynchronizationBarriers
    vectorlm_fn: Callable[[SynchronizationBarriers, int], None]

    def launch_vectorlm(self) -> None:
        """Launch vectorlm logic in a separate thread.

        Params:
        ------
             vectorlm_fn: VectorLM logic. Requires no argument. Be sure to
                populate all arguments via functools.partial.
            barriers: SynchronizationBarriers for synchronizing VectorLM
                and vLLM access to NCCL.
        """
        assert hasattr(self, "barriers")
        assert hasattr(self, "vectorlm_fn")


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

        import torch.distributed

        rank = torch.distributed.get_rank()

        # placeholder for output value,
        # populate on rank 0 and then broadcast.
        # torch requires placeholder element in object list.
        output = [None]
        if rank == 0:
            output = [fn(*args, **kwargs)]

        # fn might access torch.dist, which might conflict with
        # broadcast_object_list. Hence, keep all ranks witing until fn returns.
        # on rank 0.
        barriers.before_generation.wait()

        torch.distributed.broadcast_object_list(output)
        return output[0]

    return _wrapped_fn  # type: ignore[]
