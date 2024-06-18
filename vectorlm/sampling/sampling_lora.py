from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch.distributed as dist
import vllm
from vllm.lora.request import LoRARequest

from vectorlm.utils.save_utils import save_peft_adapter

from .abstract import AbstractSamplingEngine
from .utils import SynchronizationBarriers, multiprocess_wrap

if TYPE_CHECKING:
    from peft.peft_model import PeftModel


class LoRASamplingEngine(AbstractSamplingEngine):
    """Sampling engine optimized for LoRA PEFT."""

    def __init__(
        self,
        vllm_llm: vllm.LLM | None = None,
        sampling_params: vllm.SamplingParams | None = None,
        synchronization_barriers: SynchronizationBarriers | None = None,
        adapter_temp_folder: str | None = None,
    ) -> None:
        """Initialize sampling engine.

        Params:
            vllm_llm: Instance of vllm.LLM, required only for rank 0.
            sampling_params: Optionally, specify default sampling params.
            adapter_temp_folder: Temporary path where temporary adapter weights
              are saved. If not specified, f`/dev/shm/{job_id}`
        """
        assert synchronization_barriers is not None
        self.barriers = synchronization_barriers
        self.sampling_params = sampling_params

        if adapter_temp_folder is not None:
            self.adapter_temp_folder = adapter_temp_folder
        else:
            slurm_job_id_or_placeholder = os.environ.get("SLURM_JOB_ID", "0")

            # Manually specify the in-memory /dev/shm filesystem
            # to avoid disk wear and overhead.
            self.adapter_base_folder = "/dev/shm/"  # noqa: S108
            self.adapter_temp_folder = os.path.join(
                self.adapter_base_folder,
                slurm_job_id_or_placeholder,
            )

        if dist.get_rank() == 0:
            assert vllm_llm is not None
            self.vllm_llm = vllm_llm
            generate_fn_raw = vllm_llm.generate
        else:
            # placeholder, as the wrapped_fn won't be invoked outside rank-0.
            generate_fn_raw = None

        self.generate_fn = multiprocess_wrap(generate_fn_raw, self.barriers)
        self.vllm_train_step = -1

    def update(self, model: PeftModel, train_step: int) -> None:
        """Update model in sampling engine if the current copy is stale.

        Params:
            model: PeftModel, up-to-date model
            train_step: int, train step of the given model.
        """
        self.barriers.before_generation.wait()
        if self.vllm_train_step != train_step:
            save_peft_adapter(model, self.adapter_temp_folder)
            self.vllm_train_step = train_step
            self.lora_request = LoRARequest(
                "_vectorlm",
                self.vllm_train_step + 1,
                self.adapter_temp_folder,
            )

        self.barriers.after_generation.wait()

    def generate(
        self,
        prompts: list[str],
        sampling_params: vllm.SamplingParams | None = None,
    ) -> list[vllm.RequestOutput]:
        """Generate continuation for the given prompts. Invoke at all ranks.

        Output will be broadcasted to all ranks.

        Params:
        ------
            prompts: List of input prompts.
            sampling_params: Optionally, override self.sampling_params in
                this request only.

        Returns
        -------
            Output from vllm: list[vllm.RequestOutput], one for each prompt.

        """
        return_value = self.generate_fn(
            prompts,
            sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        assert len(return_value) == len(prompts)

        return return_value
