from __future__ import annotations

import os

import torch
import torch.distributed as dist
import vllm
from vllm.lora.request import LoRARequest

from vectorlm.trainer import Trainer
from vectorlm.utils.save_utils import save_peft_adapter

from .abstract import AbstractSamplingEngine
from .utils import SynchronizationBarriers


class LoRASamplingEngine(AbstractSamplingEngine):
    """Sampling engine optimized for LoRA PEFT."""

    def __init__(
        self,
        trainer: Trainer,
        vllm_llm: vllm.LLM | None = None,
        sampling_params: vllm.SamplingParams | None = None,
        synchronization_barriers: SynchronizationBarriers | None = None,
        adapter_temp_folder: str | None = None,
    ) -> None:
        """Initialize sampling engine.

        Params:
            trainer: Trainer instance.
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

        # Trigger FSDP initialization before retrieving weights.
        # Otherwise FSDP is_root flag might be set incorrectly.
        _wrapped_model = trainer.model
        assert _wrapped_model is not None
        _wrapped_model(input_ids=torch.zeros((1, 1), dtype=torch.int))
        self.vllm_train_step = -1

        self.update(trainer)

    def update(self, trainer: Trainer | None = None) -> None:
        """Inform the sampling engine that the model in trainer is updated.

        Params:
            trainer: Optionally, replace self.trainer with the provided value.
        """
        if trainer is not None:
            self.trainer = trainer

        wrapped_model = self.trainer.model
        assert wrapped_model is not None

        if self.vllm_train_step != self.trainer.tr_step:
            save_peft_adapter(wrapped_model, self.adapter_temp_folder)
            assert self.trainer.tr_step is not None
            assert self.trainer.tr_step >= 0
            self.vllm_train_step = self.trainer.tr_step
            self.lora_request = LoRARequest(
                "_vectorlm",
                self.vllm_train_step + 1,
                self.adapter_temp_folder,
            )

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
        # placeholder for output value,
        # populate on rank 0 and then broadcast.
        return_value_local: list[vllm.RequestOutput] | list[None]
        self.barriers.before_generation.wait()

        if dist.get_rank() == 0:
            assert self.vllm_train_step is not None
            return_value_local = self.vllm_llm.generate(
                prompts,
                sampling_params,
                lora_request=self.lora_request,
                use_tqdm=True,
            )
            assert len(return_value_local) == len(prompts)

        else:
            # torch requires placeholder output lists of same length as src.
            return_value_local = [None] * len(prompts)

        self.barriers.after_generation.wait()

        dist.broadcast_object_list(return_value_local)
        return_value: list[vllm.RequestOutput] = []
        for broadcasted_item in return_value_local:
            assert broadcasted_item is not None
            return_value.append(broadcasted_item)

        return return_value
