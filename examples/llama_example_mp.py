"""llama_example, but uses multiprocessing in place of torchrun"""

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

mp = multiprocessing.get_context("fork")


class _VLLMCallbackWrapper:
    """Provide vLLM Engine access to multiprocess.Process workers.

    vLLM engine is initialized only after the initialize_engine call.
    """

    def __init__(
        self,
        non_driver_workers: list[VectorLMWorker],
        engine_config: EngineConfig,
        vectorlm_config: Config,
        world_size: int,
    ) -> None:
        """Instantiate class without initializing wrapped vLLM engine."""
        self.llm_engine: LLMEngine | None = None
        self.llm: LLM | None = None
        self.non_driver_workers = non_driver_workers
        self.engine_config = engine_config

        # torch.dist init barrier for rank 0 vectorlm process.
        # ensures rank 0 vectorlm achieves torch.dist
        # before starting rank 0 Worker.
        self.root_vectorlm_dist_init_barrier = threading.Barrier(2)
        self.vectorlm_main_fn = partial(
            main,
            vectorlm_config,
            0,
            world_size,
            self.root_vectorlm_dist_init_barrier,
        )

    def initialize_engine(self) -> None:
        """Initialize vLLM engine.

        Invoke this method only after vLLM workers are all ready.
        """
        ManagedMultiProcGPUExecutor.workers = tuple(
            self.non_driver_workers,
        )
        ManagedMultiProcGPUExecutor.vectorlm_main_fn = self.vectorlm_main_fn
        ManagedMultiProcGPUExecutor.vectorlm_dist_init_barrier = (
            self.root_vectorlm_dist_init_barrier
        )

        self.llm_engine = LLMEngine(
            **self.engine_config.to_dict(),
            executor_class=ManagedMultiProcGPUExecutor,
            log_stats=False,
        )
        self.llm = ManagedLLM(self.llm_engine)
        print(f"Instantiated ManagedLLM: {self.llm}")


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
    ) -> None:
        """Instantiate LocalWorkerVllm wrapper.

        vectorlm_dist_init_barrier ensures that torch.dist is initialized in
        the vectorlm thread and not the main thread (vllm) of the process.
        """
        self.vllm_engine_config = vllm_engine_config
        self.vectorlm_dist_init_barrier = threading.Barrier(2)
        self.vectorlm_config = vectorlm_config
        self.local_rank = local_rank
        self.world_size = world_size
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
                self.vectorlm_dist_init_barrier,
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
        )
        for local_rank in range(1, world_size)
    ]
    vllm_callback_wrapper = _VLLMCallbackWrapper(
        non_driver_workers,
        vllm_engine_config,
        vectorlm_config,
        world_size,
    )

    for worker in non_driver_workers:
        worker.start()

    vllm_callback_wrapper.initialize_engine()
    assert vllm_callback_wrapper.llm is not None
    output = vllm_callback_wrapper.llm.generate("Vector Institute is")
    print(output)
