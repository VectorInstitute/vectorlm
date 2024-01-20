from __future__ import annotations

import torch
import yaml
from box import Box


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
        return batch

    def _reverse_tensor(
        self,
        x: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Tensors in `x` have shape (S,)."""
        return [t.flip(0) for t in x]
