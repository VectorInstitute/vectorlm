from __future__ import annotations

import math

import datasets
import torch
import torch.distributed as dist
import yaml
from box import Box
from torch.utils.data import DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer


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
        padding_side: A side of the sequence that gets padded.
    """

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int,
        max_seq_len: int,
        padding_side: str,
    ) -> None:
        """Initialize the data collator instance.

        Args:
        ----
            pad_token_id: The token id used for padding.
            ignore_index: The index used to ignore labels while calculating
                loss.
            max_seq_len: The maximum sequence length to expect.
            padding_side: The side of the sequence which is padded.
        """
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len
        self.padding_side = padding_side

    def __call__(
        self,
        instances: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        """Create an input batch.

        Convert incoming tokenized instances to the relevant batch items. The
        batch contains `id` which is the unique id for a training data point,
        `input_ids` which are tokens in the sequence with padding applied,
        `labels` which are the predicted tokens with ignore index applied,
        and `attention_mask` which dictates which tokens will be masked out
        during the self-attention mechanism.

        Args:
        ----
            instances: A list containing dictionary datapoints in a HF format.

        Returns:
        -------
            A dictionary containing a batch that we can input to our model.
        """
        batch = {}
        keys = ["input_ids", "labels"]
        input_ids, labels = tuple([
            torch.tensor(
                instance[key][0:self.max_seq_len],
            ) for instance in instances] for key in keys
        )
        batch["id"] = torch.tensor(
            [instance["id"] for instance in instances],
        )

        if self.padding_side == "left":
            input_ids = self._reverse_tensor(input_ids)
            labels = self._reverse_tensor(labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index,
        )

        if self.padding_side == "left":
            input_ids = input_ids.flip(dims=(1,))
            labels = labels.flip(dims=(1,))

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] = batch["input_ids"].ne(self.pad_token_id)
        # print(batch["attention_mask"])
        return batch

    def _reverse_tensor(
        self,
        x: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Tensors in `x` have shape (S,)."""
        return [t.flip(0) for t in x]


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
        self.setup_dataloaders()

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
