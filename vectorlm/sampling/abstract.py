"""Wrapper around vLLM. Also handles synchronization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import vllm

if TYPE_CHECKING:
    import torch
    from vectorlm.trainer import Trainer

    from .utils import SynchronizationBarriers


class AbstractSamplingEngine(ABC):
    """Interface for the sampling engine."""

    def __init__(
        self,
        vllm_llm: vllm.LLM | None = None,
        sampling_params: vllm.SamplingParams | None = None,
        synchronization_barriers: SynchronizationBarriers | None = None,
    ) -> None:
        """Initialize sampling engine.

        Params:
            vllm_llm: Instance of vllm.LLM, required only for rank 0.
            sampling_params: Optionally, specify default sampling params.
            synchronization_barriers: Optionally, supply barriers to
                prevent workers from accessing GPU while vLLM is running.

        """
        self.vllm_llm = vllm_llm
        self.sampling_params = sampling_params
        self.synchronization_barriers = synchronization_barriers
        self.vllm_train_step = -1

    @abstractmethod
    def update(self, model: torch.nn.Module, train_step: int) -> None:
        """Update model in sampling engine if the current copy is stale.

        Params:
            model: PeftModel, up-to-date model
            train_step: int, train step of the given model.
        """
        if self.vllm_train_step != train_step:
            # Update parameters of self.vllm_llm using the given `model``.
            return

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        sampling_params: vllm.SamplingParams | None = None,
    ) -> list[vllm.RequestOutput]:
        """Generate continuation for the given prompts synchronously.

        Invoke at all ranks. Output will be broadcasted to all ranks.

        Params:
        ------
            prompts: List of input prompts.
            sampling_params: Optionally, override self.sampling_params in
                this request only.

        Returns
        -------
            Output from vllm: list[vllm.RequestOutput], one for each prompt.

        """
