"""llama_example, but uses multiprocessing in place of torchrun.

Each non-rank-0 worker process should spawn vectorlm logic in a
separate thread (but same process) but won't run the actual
vectorlm logic until the vLLM Engine is initialized- inference
weights loaded into each worker.

To do so without duplicating vLLM code, observe that only the main process
(rank 0) is aware that vLLM engine was initialized properly
(when LLMEngine.__init__ returns.) Hence, one way to implement this
setup is to block the vectorlm thread with a multiprocessing synchronization
feature (e.g., a Barrier shared across all processes) that the rank 0 process
can remotely unblock.

Edit: It seems that vllm.entrypoint.llm.LLM generate calls aren't
entirely blocking.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import multiprocessing.context
import multiprocessing.managers
import threading
from functools import partial
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vllm.worker.worker_base import WorkerBase

from llama_example import main
from vllm.engine.arg_utils import EngineArgs, EngineConfig
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.local_worker_utils import LocalWorkerVllm, ResultHandler
from vllm.entrypoints.llm import LLM
from vllm.worker.worker import init_worker_distributed_environment

from vectorlm.sampling.utils import (
    ManagedLLM,
    ManagedMultiProcGPUExecutor,
    _ensure_torch_dist_is_initialized,
    _get_rdvz_url,
    get_vllm_worker_factory,
)
from vectorlm.utils.data_utils import Config

logging.basicConfig(level=logging.DEBUG)


class _VLLMCallbackWrapper:
    """Provide vLLM Engine access to multiprocess.Process workers.

    vLLM engine is initialized only after the initialize_engine call.
    """

    def __init__(
        self,
        non_driver_workers: list[VectorLMWorker],
        vllm_result_handler: ResultHandler,
        engine_config: EngineConfig,
        vectorlm_config: Config,
        world_size: int,
        vllm_init_barrier: threading.Barrier,
    ) -> None:
        """Instantiate class without initializing wrapped vLLM engine."""
        self.llm_engine: LLMEngine | None = None
        self.llm: LLM | None = None
        self.non_driver_workers = non_driver_workers
        self.vllm_result_handler = vllm_result_handler
        self.engine_config = engine_config
        self.vllm_init_barrier = vllm_init_barrier

        # Might not be required since LLM.generate is blocking.
        # torch.dist.barrier might be sufficient for blocking
        # other worker processes.
        self.gpu_access_lock = threading.Lock()

        self.vectorlm_main_fn = partial(
            main,
            vectorlm_config,
            0,
            world_size,
            self.vllm_init_barrier,
            self.get_engine,
        )

    def initialize_engine(self) -> None:
        """Initialize vLLM engine.

        Invoke this method only after vLLM workers are all ready.
        """
        ManagedMultiProcGPUExecutor.workers = tuple(
            self.non_driver_workers,
        )
        ManagedMultiProcGPUExecutor.vectorlm_main_fn = self.vectorlm_main_fn
        ManagedMultiProcGPUExecutor.result_handler = self.vllm_result_handler

        self.llm_engine = LLMEngine(
            **self.engine_config.to_dict(),
            executor_class=ManagedMultiProcGPUExecutor,
            log_stats=False,
        )

        self.llm = ManagedLLM(self.llm_engine)
        print(f"Instantiated ManagedLLM: {self.llm}")

    def get_engine(self) -> LLM:
        """Return LLM instance.

        Invoke this method only within the main (rank 0 driver) process.
        """
        if self.llm is None:
            self.initialize_engine()

        llm = self.llm
        assert llm is not None
        return llm

    def join_vectorlm_thread(self) -> None:
        """Join the rank 0 (main process) vectorlm thread.

        Invoke this function only after initialize_engine.
        """
        assert self.llm_engine is not None
        model_executor = self.llm_engine.model_executor
        assert isinstance(model_executor, ManagedMultiProcGPUExecutor)

        model_executor.vectorlm_thread.join()


class VectorLMWorker(LocalWorkerVllm):
    """Worker for running VectorLM logic alongside vLLM worker.

    Important: do not use this instance for the rank 0 (root) process.

    Note that nccl requires that only one process may have access
    to each GPU. Each LocalWorkerVllm is a multiprocessing.Process.
    Vectorlm logic would be launched as a thread within each of these
    proceses.

    Spawn no more than one such instance for each GPU.
    """

    def __init__(
        self,
        result_handler: ResultHandler,
        worker_factory: Callable[[], WorkerBase],
        vllm_engine_config: EngineConfig,
        vectorlm_config: Config,
        local_rank: int,
        world_size: int,
        vllm_init_barrier: threading.Barrier,
    ) -> None:
        """Instantiate LocalWorkerVllm wrapper.

        vectorlm_dist_init_barrier ensures that torch.dist is initialized in
        the vectorlm thread and not the main thread (vllm) of the process.
        """
        self.vllm_engine_config = vllm_engine_config
        self.gpu_access_lock = threading.Lock()
        self.vectorlm_config = vectorlm_config
        self.local_rank = local_rank
        self.world_size = world_size
        self.vllm_init_barrier = vllm_init_barrier

        self.vllm_init_callback: Callable[[], None] | None = None

        super().__init__(result_handler, worker_factory)

    def run(self) -> None:
        """Launch vectorlm logic in a separate thread."""
        print(f"rank {self.local_rank}: init_worker_dist started")
        init_worker_distributed_environment(
            self.vllm_engine_config.parallel_config,
            self.local_rank,
            _get_rdvz_url(),
            self.local_rank,
        )
        print(f"rank {self.local_rank}: init_worker_dist completed")

        _ensure_torch_dist_is_initialized()

        self.vectorlm_thread = threading.Thread(
            target=main,
            args=(
                self.vectorlm_config,
                self.local_rank,
                self.world_size,
                self.vllm_init_barrier,
                self.vllm_init_callback,
            ),
            name=f"rank-{self.local_rank}/vectorlm",
        )
        self.vectorlm_thread.start()

        super().run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--yaml_path", type=str, required=True)
    args = parser.parse_args()

    world_size: int = args.world_size
    vectorlm_config = Config(yaml_path=args.yaml_path)
    sampler_config = vectorlm_config.train_parameters.sampler  # type: ignore[]
    vllm_engine_config = EngineArgs(
        model=vectorlm_config.model,  # type: ignore[]
        gpu_memory_utilization=sampler_config.get(
            "gpu_memory_utilization",
            0.35,
        ),
        tensor_parallel_size=world_size,
        dtype=sampler_config.get("vllm_dtype", "auto"),
    ).create_engine_config()

    # Block all N vectorlm threads until main process finished
    # initializing vLLM Engine.
    vllm_init_barrier = multiprocessing.Barrier(world_size + 1)
    vllm_result_handler = ResultHandler()

    # rank 0 worker runs in the __main__ process.
    # all other ranks use one process each.
    # vectorlm logic in each ranks (including rank 0) is in a separate thread.
    non_driver_workers: list[VectorLMWorker] = [
        VectorLMWorker(
            vllm_result_handler,
            get_vllm_worker_factory(
                vllm_engine_config,
                _get_rdvz_url(),
                local_rank,
            ),
            vllm_engine_config,
            vectorlm_config,
            local_rank,
            world_size=world_size,
            vllm_init_barrier=vllm_init_barrier,
        )
        for local_rank in range(1, world_size)
    ]
    vllm_callback_wrapper = _VLLMCallbackWrapper(
        non_driver_workers,
        vllm_result_handler,
        vllm_engine_config,
        vectorlm_config,
        world_size,
        vllm_init_barrier,
    )

    for worker in non_driver_workers:
        worker.start()

    vllm_callback_wrapper.initialize_engine()
    assert vllm_callback_wrapper.llm is not None
    print("main: vllm_init_barrier waiting")
    vllm_init_barrier.wait()
    print("main: vllm_init_barrier cleared")

    vllm_init_barrier.wait()

    output = vllm_callback_wrapper.llm.generate("Vector Institute is")
    print(output)

    vllm_callback_wrapper.join_vectorlm_thread()
