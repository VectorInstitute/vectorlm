"""Generic utils for the sampling engines."""

from __future__ import annotations

import json
import os
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, TypeVar

from vllm import LLM
from vllm.engine.local_worker_utils import (
    LocalWorkerVllm,
    ResultHandler,
    WorkerMonitor,
)
from vllm.executor.multiproc_gpu_executor import (
    MultiProcGPUExecutor,
    _create_worker,
)
from vllm.utils import (
    Counter,
    get_distributed_init_method,
    set_cuda_visible_devices,
)
from vllm.worker.worker import init_worker_distributed_environment

from .abstract import AbstractSamplingEngine

if TYPE_CHECKING:
    from threading import Barrier

    from vllm import LLMEngine, SamplingParams
    from vllm.engine.arg_utils import EngineConfig
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


def _ensure_torch_dist_is_initialized() -> None:
    import torch.distributed

    assert torch.distributed.is_initialized()


def _get_rdvz_url() -> str:
    """Obtain rendezvous url for Torch dist."""
    return get_distributed_init_method(
        os.environ.get("MASTER_ADDR", "127.0.0.1"),
        int(os.environ["MASTER_PORT"]),
    )


class ManagedMultiProcGPUExecutor(MultiProcGPUExecutor):
    """MultiProcGPUExecutor, but with worker processes instantiated outside."""

    workers: tuple[LocalWorkerVllm, ...] | None = None
    vectorlm_main_fn: Callable[[], None] | None = None
    result_handler: ResultHandler | None = None

    def _init_executor(self) -> None:
        """Initialize executor without initializing workers.

        Same as MultiProcGPUExecutor but assumes self.workers is already set.

        Mostly reproduced from
        vllm/vllm-ray-optional/vllm/executor/multiproc_gpu_executor.py
        """
        assert (
            not self.speculative_config
        ), "Speculative decoding not yet supported for MultiProcGPU backend."

        # Create the parallel GPU workers.
        world_size = self.parallel_config.tensor_parallel_size
        assert self.workers is not None
        assert len(self.workers) == world_size - 1, (
            f"non-driver workers len(self.workers): {len(self.workers)} "
            f"should be (world_size - 1) {world_size - 1}"
        )

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            set_cuda_visible_devices(range(world_size))

        from torch.cuda import device_count

        assert (
            world_size <= device_count()
        ), "please set tensor_parallel_size to less than max local gpu count"

        assert self.result_handler is not None
        self.worker_monitor = WorkerMonitor(
            list(self.workers),
            self.result_handler,
        )
        self.result_handler.start()
        self.worker_monitor.start()

        distributed_init_method = _get_rdvz_url()

        # driver worker is of rank 0
        print("driver worker: init_worker_dist started")
        init_worker_distributed_environment(
            self.parallel_config,
            0,
            distributed_init_method,
            0,
        )
        print("driver worker: init_worker_dist completed")
        _ensure_torch_dist_is_initialized()

        # start vectorlm logic in the same Python process
        # (albeit in a separate thread)
        self.vectorlm_thread = threading.Thread(
            target=self.vectorlm_main_fn,
            name="driver/vectorlm",
        )
        self.vectorlm_thread.start()

        self._init_driver_worker_and_model(0, 0, distributed_init_method)


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


def get_vllm_worker_factory(
    engine_config: EngineConfig,
    distributed_init_method: str,
    rank: int,
) -> Callable[[], WorkerBase]:
    """Initialize vLLM worker."""
    return partial(
        _create_worker,
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
        cache_config=engine_config.cache_config,
        local_rank=rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        lora_config=engine_config.lora_config,
        vision_language_config=engine_config.vision_language_config,
        tensorizer_config=engine_config.tensorizer_config,
    )


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
