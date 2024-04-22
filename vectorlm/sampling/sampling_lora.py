from __future__ import annotations

import os

import torch
import torch.distributed as dist
import vllm
from vllm.lora.request import LoRARequest

from vectorlm.trainer import Trainer
from vectorlm.utils.save_utils import save_peft_adapter

from .abstract import AbstractSamplingEngine


class LoRASamplingEngine(AbstractSamplingEngine):
    """Sampling engine optimized for LoRA PEFT."""

    def __init__(
        self,
        trainer: Trainer,
        sampling_params: vllm.SamplingParams | None = None,
        base_model_name: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.3,
        adapter_temp_folder: str | None = None,
    ) -> None:
        """Initialize sampling engine.

        Params:
            trainer: Trainer instance.
            sampling_params: Optionally, specify default sampling params.
            base_model_name: Path or HuggingFace repo name of base model.
            tensor_parallel_size: Forwarded to vllm.LLM.
            gpu_memory_utilization: Forwarded to vllm.LLM.
            adapter_temp_folder: Temporary path where temporary adapter weights
              are saved. If not specified, f`/dev/shm/{job_id}`
        """
        if dist.get_rank() != 0:
            return

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

        assert (
            base_model_name is not None
        ), "base_model_name is required when instantiating LoRASamplingEngine."

        self.vllm_llm = vllm.LLM(
            base_model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
        )

        # Trigger FSDP initialization before
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
        if dist.get_rank() != 0:
            return

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
    ) -> list[list[vllm.CompletionOutput]]:
        """Generate continuation for the given prompts. Invoke only on rank 0.

        Params:
        ------
            prompts: List of input prompts.
            sampling_params: Optionally, override self.sampling_params in
                this request only.

        Returns
        -------
            Output from vllm: list[list[vllm.CompletionOutput]]
                outer layer: one for each prompt.
                inner layer: one for each output option for the prompt.

        """
        if dist.get_rank() != 0:
            msg = "LoRA sampling engine is supported only on rank 0."
            raise RuntimeError(msg)

        assert self.vllm_train_step is not None
        output_list = self.vllm_llm.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request,
            use_tqdm=True,
        )
        return [output.outputs for output in output_list]
