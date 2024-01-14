from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from transformers import get_scheduler


class PlateaeuWithWarmup(ReduceLROnPlateau):
    """Class to implement plataeu scheduling with warmup.

    Attributes
    ----------
        optimizer: An optimizer that we are adjusting the LR of.
        factor: A float factor by which we multiplicatively reduce LR by.
        patience: An integer number of steps to wait before reducing LR.
        threshold: A float threshold on the change in metric to focus only on
            significant changes.
        threshold_mode: A string, of either `rel` or `abs`, to calculate the
            current best performing threshold value. See PyTorch documentation
            for more information.
        cooldown: An integer number of steps to wait after LR has been reduced
            before resuming to continue checking to reduce LR.
        min_lr: A float determining the minimum LR to reduce to.
        eps: A float where if the difference between the old and new LR is less
            than `eps`, the update is ignored.
        verbose: A boolean determining whether to print a message to stdout
            every update.
        num_warmup_steps: An integer number of warm up steps to the maximum LR.
            The maximum LR is determined by the number of warmup steps and the
            current step.
        base_lrs: A list of base LRs for the optimizer's param groups.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
        num_warmup_steps: int = 0,
    ) -> None:
        """Initialize the PlateaeuWithWarmup scheduler class.

        Arguments:
        ---------
            optimizer: The optimizer we are adjusting the LR of.
            factor: The factor by which we reduce the LR by.
            patience: The number of steps to wait before reducing LR again.
            threshold: The threshold by which the metric must change by before
                we reduce the LR.
            threshold_mode: Either `rel` or `abs`. See class description.
            cooldown: The number of steps to wait after reducing LR before
                resuming regular operations.
            min_lr: The minimum LR we are allowed to reduce to.
            eps: The change in the new LR and old LR must be at least eps
                otherwise the update is ignored.
            verbose: Whether to print messages to stdout every LR update.
            num_warmup_steps: The number of steps we warmup the LR for.
        """
        super().__init__(
            optimizer=optimizer,
            factor=factor,
            threshold=threshold,
            threshold_mode=threshold_mode,
            eps=eps,
            verbose=verbose,
            min_lr=min_lr,
            patience=patience,
            cooldown=cooldown,
        )
        self.num_warmup_steps = num_warmup_steps
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Step the scheduler once.

        Arguments:
        ---------
            metrics: The metric we are using to measure change in LR.
            epoch: The current step.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        new_lr = None
        if epoch <= self.num_warmup_steps:

            ratio = float(epoch) / float(self.num_warmup_steps)
            new_lr = [ratio * lr for lr in self.base_lrs]
            self._reduce_lr(epoch, new_lr)
        else:

            current = float(metrics)
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch: int, new_lr: float | None = None) -> None:
        if new_lr:
            for i, (lr, param_group) in enumerate(
                zip(new_lr, self.optimizer.param_groups),
            ):
                param_group["lr"] = lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                    "%.5d") % epoch
                    print(f"Epoch {epoch_str}: reducing learning rate"
                        f" of group {i} to {new_lr:.4e}.")
        else:
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group["lr"])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group["lr"] = new_lr
                    if self.verbose:
                        epoch_str = ("%.2f" if isinstance(epoch, float) else
                                    "%.5d") % epoch
                        print(f"Epoch {epoch_str}: reducing learning rate"
                            f" of group {i} to {new_lr:.4e}.")

    def get_last_lr(self) -> list[float]:
        """Return the current LRs for each optimizer param group."""
        return [group["lr"] for group in self.optimizer.param_groups]


def get_custom_scheduler(
    name: str,
    *args: list[Any],
    **kwargs: dict[str, Any],
) -> LRScheduler | ReduceLROnPlateau:
    """Return the LR scheduler.

    Return either the custom ReduceLROnPlateau scheduler, or one
    implemented by HF. See `get_scheduler` in HF `transformers` for acceptable
    options.

    Args:
    ----
        name: The name of the scheduler
        args: The scheduler specific args.
        kwargs: The scheduler specific kwargs.
    """
    if name == "plataeu-with-warmup":
        scheduler = PlateaeuWithWarmup(*args, **kwargs)
        # required, otherwise the very first step the optimizer takes is at the
        # maximum set LR (because we step the scheduler *after* we step the
        # optimizer. As a result, the optimizer is set to max LR on the first
        # iteration.)
        scheduler.step(None)
        return scheduler
    return get_scheduler(name, *args, **kwargs)
