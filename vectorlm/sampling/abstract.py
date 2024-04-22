"""Wrapper around sampling engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

import vllm

from vectorlm.trainer import Trainer


class AbstractSamplingEngine(ABC):
    """Interface for the sampling engine."""

    def __init__(
        self,
        trainer: Trainer,
        sampling_params: vllm.SamplingParams | None = None,
    ) -> None:
        """Initialize sampling engine.

        Params:
            trainer: Trainer instance.
            sampling_params: Optionally, specify default sampling params.

        """
        self.trainer = trainer
        self.sampling_params = sampling_params

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
    ) -> list[list[vllm.CompletionOutput]]:
        """Generate continuation for the given prompts synchronously.

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
