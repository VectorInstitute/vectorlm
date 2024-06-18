"""Supply LoRASamplingEngine to llama_example.

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

See docs.google.com/presentation/d/1FCa5O8RYYkRRCAAcXhqCvomePo5fEfhjQciSteTEJ30
for more detail on this architecture.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

from llama_example import main
from vllm import EngineArgs
from vllm.executor.multiproc_worker_utils import ResultHandler, mp

from vectorlm.sampling import (
    LoRASamplingEngine,
    SamplingEngineProvider,
    SynchronizationBarriers,
)
from vectorlm.utils.data_utils import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--yaml_path", type=str, required=True)
    args = parser.parse_args()

    world_size: int = args.world_size
    vectorlm_config = Config(yaml_path=args.yaml_path)
    sampler_config = vectorlm_config.train_parameters.sampler  # type: ignore[reportAttributeAccessIssue]
    vllm_engine_config = EngineArgs(
        model=vectorlm_config.model,  # type: ignore[reportAttributeAccessIssue]
        gpu_memory_utilization=sampler_config.get(
            "gpu_memory_utilization",
            0.35,
        ),
        tensor_parallel_size=world_size,
        dtype=sampler_config.get("vllm_dtype", "auto"),
        enable_lora=True,
    ).create_engine_config()
    os.environ["WORLD_SIZE"] = str(world_size)

    # Block all N vectorlm threads until main process finished
    # initializing vLLM Engine. Additionally, block vectorlm
    # threads as long as vLLM tasks are running.
    barriers = SynchronizationBarriers(
        # (n+1) threads: __main__, and n vectorlm threads (including main).
        vllm_init=mp.Barrier(world_size + 1),
        # n vectorlm threads.
        before_generation=mp.Barrier(world_size),
        after_generation=mp.Barrier(world_size),
    )
    vllm_result_handler = ResultHandler()

    # rank 0 worker runs in the __main__ process.
    # all other ranks use one process each.
    # vectorlm logic in each ranks (including rank 0) is in a separate thread
    # from the vLLM worker logic.
    vllm_callback_wrapper = SamplingEngineProvider(
        vllm_engine_config,
        barriers,
        LoRASamplingEngine,
        partial(main, vectorlm_config),
    )

    vllm_callback_wrapper.initialize_engine()
    assert vllm_callback_wrapper.llm is not None
    output = vllm_callback_wrapper.llm.generate("Vector Institute is")
    print(output[0].prompt + output[0].outputs[0].text)

    vllm_callback_wrapper.join_vectorlm_thread()
