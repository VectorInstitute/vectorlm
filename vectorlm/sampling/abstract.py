"""Wrapper around sampling engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import vllm

if TYPE_CHECKING:
    from vectorlm.trainer import Trainer

    from .utils import SynchronizationBarriers


class AbstractSamplingEngine(ABC):
    """Interface for the sampling engine."""

    def __init__(
        self,
        trainer: Trainer,
        vllm_llm: vllm.LLM | None = None,
        sampling_params: vllm.SamplingParams | None = None,
        synchronization_barriers: SynchronizationBarriers | None = None,
    ) -> None:
        """Initialize sampling engine.

        Params:
            trainer: Trainer instance.
            vllm_llm: Instance of vllm.LLM, required only for rank 0.
            sampling_params: Optionally, specify default sampling params.
            synchronization_barriers: Optionally, supply barriers to
                prevent workers from accessing GPU while vLLM is running.

        """
        self.trainer = trainer
        self.vllm_llm = vllm_llm
        self.sampling_params = sampling_params
        self.synchronization_barriers = synchronization_barriers

    def update(self, trainer: Trainer | None = None) -> None:
        """Inform the sampling engine that the model in trainer is updated.

        Params:
            trainer: Optionally, replace self.trainer with the provided value.
        """
        if trainer is not None:
            self.trainer = trainer

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
