from __future__ import annotations

import math

import datasets
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer

from vectorlm.utils.data_utils import Config, DataCollatorWithPadding


class Dataset:
    """The dataset class.

    Attributes
    ----------
        config: A dataset config.
        tokenizer: An input tokenizer.
        original_length: An integer denoting the original length of the dataset
            prior to removing already trained examples (in the case of a
            checkpoint).
        train_dataloader: A dataloader for the train set.
        eval_dataloader: A dataloader for the test set.
        train_ds: A training dataset.
        eval_ds: An evaluation dataset.
        train_bs: A per-device batch size for training.
        eval_bs: A per-device batch size for evaluating.
        _processed_ids: A tensor of already trained examples.

    """

    def __init__(
        self,
        config: Config,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        """Initialize the dataset class.

        Args:
        ----
            config: The dataset config.
            tokenizer: The input tokenizer.

        """
        self.config = config
        self._processed_ids = torch.tensor([]).to(torch.cuda.current_device())
        self.tokenizer = tokenizer
        self.original_length = -100
        self.train_bs = config.train_bs
        self.eval_bs = config.eval_bs
        self.train_ds = None
        self.eval_ds = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.load_datasets()

    def reset_dataloaders(self) -> None:
        """Reset dataloaders."""
        self._processed_ids = torch.tensor([]).to(torch.cuda.current_device())

    def update_processed_ids(self, new_ids: torch.Tensor) -> None:
        """Update processed ids with an incoming stream of ids."""
        self._processed_ids = torch.cat([self._processed_ids, new_ids])

    def get_processed_ids(self) -> torch.Tensor:
        """Return the current processed ids."""
        return self._processed_ids

    def set_processed_ids(self, ids: list[int]) -> None:
        """Set the checkpointed processed ids."""
        self._processed_ids = torch.tensor(ids).to(torch.cuda.current_device())

    def load_datasets(self) -> None:
        """Load datasets into memory."""
        dirs_passed = self.config.get("train_ds", "") and \
            self.config.get("eval_ds", "")

        if not dirs_passed:
            msg = "`train_ds` and `eval_ds` are missing from config."
            raise KeyError(msg)
        self.train_ds = datasets.load_from_disk(self.config.train_ds)
        self.eval_ds = datasets.load_from_disk(self.config.eval_ds)
        self.original_length = math.ceil(len(self.train_ds) / self.train_bs)

    def setup_dataloaders(self) -> None:
        """Load the data and create the dataloaders."""
        # filter out unwanted rows from data we've already trained on
        if len(self._processed_ids):
            to_remove = set(self._processed_ids.int().tolist())
            self.train_ds = self.train_ds.filter(
                lambda example: _remove_unwanted_rows(example, to_remove),
                batched=True,
                num_proc=4,
                batch_size=5000,
            )

        if dist.get_rank() == 0:
            print(f"Train dataset length {len(self.train_ds)}")
            print(f"Eval dataset length {len(self.eval_ds)}")

        dc = DataCollatorWithPadding(
            self.tokenizer.pad_token_id,
            self.config.ignore_index,
            self.tokenizer.model_max_length,
            self.tokenizer.padding_side,
        )

        train_sampler = DistributedSampler(
            self.train_ds,
            dist.get_world_size(),
            dist.get_rank(),
            shuffle=True,
        )
        test_sampler = DistributedSampler(
            self.eval_ds,
            dist.get_world_size(),
            dist.get_rank(),
            shuffle=False,
        )

        self.train_dataloader = DataLoader(
            self.train_ds,
            collate_fn=dc,
            batch_size=self.train_bs,
            sampler=train_sampler,
            shuffle=False,
        )
        self.eval_dataloader = DataLoader(
            self.eval_ds,
            collate_fn=dc,
            batch_size=self.eval_bs,
            sampler=test_sampler,
            shuffle=False,
        )


def _remove_unwanted_rows(
    examples: datasets.Dataset,
    rows: list(int),
) -> datasets.Dataset:
    ids = examples["id"]
    assertion_lst = []
    for idx in ids:
        if idx in rows:
            assertion_lst.append(False)
        else:
            assertion_lst.append(True)
    assert len(assertion_lst) == len(
        ids,
    ), f"Length of assertion list is {len(assertion_lst)}, expected {len(ids)}"
    return assertion_lst
