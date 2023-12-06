from __future__ import annotations

import torch
from box import Box
import jsonpickle
import os
import yaml
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch.distributed as dist
import datasets
import math
from torch.utils.data import DataLoader, DistributedSampler
import random


class Config:
    """Config file for model training.

    Attributes
    ----------
        yaml_path: A path to the yaml file that stores our config.
        to_box: A boolean indicating whether to box our config.
    """

    def __init__(self, yaml_path: str, to_box: bool = True) -> None:
        """Initialize the config instance.

        Args:
        ----
            yaml_path: The string path to the config yaml.
            to_box: Defines whether this initialization will use dot notation.
        """
        self.yaml_path = yaml_path
        self.to_box = to_box
        self.load_yaml()

    def _to_box(self) -> None:
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                self.__setattr__(key, Box(val))

    def load_yaml(self) -> None:
        """Load the config yaml and convert to box format if needed."""
        with open(self.yaml_path) as in_path:
            _config = yaml.safe_load(in_path)

        for k,v in _config.items():
            self.__setattr__(k, v)
        if self.to_box:
            self._to_box()


class DataCollatorWithPadding:
    """Data collator for preprocessing a batch.

    Similar to the one offered by HF, but here we can keep track of any extra
    keys. In particular, the "id" key is used for dataloader checkpointing and
    we would want to keep this in the batch.

    Attributes
    ----------
        pad_token_id: A token id that is used for padding.
        ignore_index: A value used for ignoring a given token in labels.
        max_seq_len: An integer denoting the maximum sequence length.
    """

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int,
        max_seq_len: int,
    ) -> None:
        """Initialize the data collator instance.

        Args:
        ----
            pad_token_id: The token id used for padding.
            ignore_index: The index used to ignore labels while calculating
                loss.
            max_seq_len: The maximum sequence length to expect.
        """
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len

    def __call__(
        self,
        instances: dict[str, list[int]]
    ) -> dict[str, torch.Tensor]:
        """Create an input batch.

        Convert incoming tokenized instances to the relevant batch items. The
        batch contains `id` which is the unique id for a training data point,
        `input_ids` which are tokens in the sequence with padding applied,
        `labels` which are the predicted tokens with ignore index applied,
        and `attention_mask` which dictates which tokens will be masked out
        during the self-attention mechanism.

        Args:
            instances: A dictionary containing preprocessed, tokenized data.

        Returns:
            A dictionary containing a batch that we can input to our model.
        """
        batch = {}
        if "id" in instances[0]:
            if "raw_data_id" in instances[0]:
                keys = ["input_ids", "labels", "raw_data_id"]
                input_ids, labels, raw_data_id = tuple(
                    [
                        torch.tensor(
                            instance[key][0:self.max_seq_len],
                        ) for instance in instances
                    ] for key in keys
                )
                batch["raw_data_id"] = raw_data_id
            else:
                keys = ["input_ids", "labels"]
                input_ids, labels = tuple(
                    [
                        torch.tensor(
                            instance[key][0:self.max_seq_len],
                        ) for instance in instances
                    ] for key in keys
                )
            batch["id"] = torch.tensor(
                [instance["id"] for instance in instances],
            )
        else:
            keys = ["input_ids", "labels"]
            input_ids, labels = tuple(
                [
                    torch.tensor(
                        instance[key][0:self.max_seq_len],
                    ) for instance in instances
                ] for key in keys
            )

        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id,
        )
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index,
        )
        batch["attention_mask"] = batch["input_ids"].ne(self.pad_token_id)
        return batch


class Dataset:
    """The dataset class.

    Attributes:
        config: A dataset config.
        processed_ids: A tensor of already trained examples.
        tokenizer: An input tokenizer.
        original_length: An integer denoting the original length of the dataset
            prior to removing already trained examples (in the case of a
            checkpoint).
        train_dataloader: A dataloader for the train set.
        eval_dataloader: A dataloader for the test set.
        train_bs: A per-device batch size for training.
        eval_bs: A per-device batch size for evaluating.
    """
    def __init__(
        self,
        config: Config,
        tokenizer: PreTrainedTokenizer,
        checkpoint_path: str = None,
    ) -> None:
        """Initialize the dataset class.
        
        Args:
            config: The dataset config.
            tokenizer: The input tokenizer.
            checkpoint_path: The path where a checkpoint is saved, if it
                exists.
        """
        self.config = config
        self.processed_ids = torch.tensor([]).to(torch.cuda.current_device())
        self.tokenizer = tokenizer
        self.original_length = -100
        self.train_dataloader = None
        self.eval_dataloader = None
        self.train_bs = config.train_bs
        self.eval_bs = config.eval_bs
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        pass
    
    def save_checkpoint(self, path: str) -> None:
        pass

    def reset_dataloaders(self) -> None:
        pass

    def update_processed_ids(self, new_ids: torch.Tensor) -> None:
        self.processed_ids = torch.cat([self.processed_ids, new_ids])
    
    def setup_dataloaders(self) -> None:
        """Loads the data and creates the dataloaders."""
        train_bs = self.config.train_bs
        eval_bs = self.config.eval_bs
        dirs_passed = self.config.get("train_ds", "") and \
            self.config.get("eval_ds", "")

        if not dirs_passed:
            raise KeyError(
                "Data directories are missing from the config."
            )
        train_ds = datasets.load_from_disk(self.config.train_ds)
        eval_ds = datasets.load_from_disk(self.config.test_ds)
        self.original_length = math.ceil(len(train_ds) / train_bs)

        # filter out unwanted rows from data we've already trained on
        if self.processed_ids:
            to_remove = set(self.processed_ids)
            train_ds = train_ds.filter(
                lambda example: _remove_unwanted_rows(example, to_remove),
                batched=True,
                num_proc=8,
                batch_size=5000,
            )

        if dist.get_rank() == 0:
            print(f"Train dataset length {len(train_ds)}")
            print(f"Eval dataset length {len(eval_ds)}")

        dc = DataCollatorWithPadding(
            self.tokenizer.pad_token_id,
            self.config.ignore_index,
            self.tokenizer.model_max_length,
        )

        train_sampler = DistributedSampler(
            train_ds,
            dist.get_world_size(),
            dist.get_rank(),
            shuffle=True,
        )
        test_sampler = DistributedSampler(
            eval_ds,
            dist.get_world_size(),
            dist.get_rank(),
            shuffle=False,
        )

        self.train_dataloader = DataLoader(
            train_ds,
            collate_fn=dc,
            batch_size=train_bs,
            sampler=train_sampler,
            shuffle=False,
        )
        self.eval_dataloader = DataLoader(
            eval_ds,
            collate_fn=dc,
            batch_size=eval_bs,
            sampler=test_sampler,
            shuffle=False,
        )


def _remove_unwanted_rows(
    examples: datasets.Dataset,
    rows: list(int)
) -> datasets.Dataset:
    ids = examples["id"]
    assertion_lst = []
    for id in ids:
        if id in rows:
            assertion_lst.append(False)
        else:
            assertion_lst.append(True)
    assert len(assertion_lst) == len(
        ids
    ), f"Length of assertion list is {len(assertion_lst)}, expected {len(ids)}"
    return assertion_lst



if __name__ == "__main__":
    breakpoint()
