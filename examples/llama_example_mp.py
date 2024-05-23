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
from functools import partial
from typing import Callable

from llama_example import main
from vllm.engine.arg_utils import EngineArgs, EngineConfig
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.llm import LLM
from vllm.executor.multiproc_worker_utils import ResultHandler, mp

from vectorlm.sampling.utils import (
    ManagedLLM,
    ManagedMultiProcGPUExecutor,
    SynchronizationBarriers,
)
from vectorlm.utils.data_utils import Config


class _VLLMCallbackWrapper:
    """Provide vLLM Engine access to multiprocess.Process workers.

    vLLM engine is initialized only after the initialize_engine call.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        vectorlm_config: Config,
        world_size: int,
        barriers: SynchronizationBarriers,
    ) -> None:
        """Instantiate class without initializing wrapped vLLM engine."""
        self.llm_engine: LLMEngine | None = None
        self.llm: LLM | None = None
        self.engine_config = engine_config
        self.barriers = barriers

        # Only missing args is local_rank.
        self.vectorlm_fn: Callable[[int], None] = partial(
            main,
            vectorlm_config,
            world_size,
            self.get_vllm_llm,
            self.barriers,
        )

    def initialize_engine(self) -> None:
        """Initialize vLLM engine.

        Invoke this method only after vLLM workers are all ready.
        """
        ManagedMultiProcGPUExecutor.vectorlm_fn = self.vectorlm_fn

        self.llm_engine = LLMEngine(
            **self.engine_config.to_dict(),
            executor_class=ManagedMultiProcGPUExecutor,
            log_stats=False,
        )

        self.llm = ManagedLLM(self.llm_engine)
        print(f"Instantiated ManagedLLM: {self.llm}")

    def get_vllm_llm(self) -> LLM:
        """Return LLM instance.

        Invoke this method only within the main (rank 0 driver) process.
        """
        assert (
            self.llm is not None
        ), "Must finish initialize_engine before starting vectorlm logic."

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
        model_executor.rank_0_vectorlm_thread.join()


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
        enable_lora=True,
    ).create_engine_config()

    # Block all N vectorlm threads until main process finished
    # initializing vLLM Engine. Additionally, block vectorlm
    # threads as long as vLLM tasks are running.
    barriers = SynchronizationBarriers(
        # (n+1) threads: __main__, and n vectorlm threads (including main).
        mp.Barrier(world_size + 1),
        # n vectorlm threads.
        mp.Barrier(world_size),
        mp.Barrier(world_size),
    )
    vllm_result_handler = ResultHandler()

    # rank 0 worker runs in the __main__ process.
    # all other ranks use one process each.
    # vectorlm logic in each ranks (including rank 0) is in a separate thread.
    vllm_callback_wrapper = _VLLMCallbackWrapper(
        vllm_engine_config,
        vectorlm_config,
        world_size,
        barriers,
    )

    vllm_callback_wrapper.initialize_engine()
    assert vllm_callback_wrapper.llm is not None
    output = vllm_callback_wrapper.llm.generate("Vector Institute is")
    print(output[0].prompt + output[0].outputs[0].text)

    print("main: vllm_init_barrier waiting")
    barriers.vllm_init.wait()
    print("main: vllm_init_barrier cleared")

    vllm_callback_wrapper.join_vectorlm_thread()
