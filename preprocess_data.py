from __future__ import annotations

import argparse
import os
from argparse import Namespace
from copy import deepcopy
from typing import Any

import datasets
from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from vectorlm.utils.data_utils import Config


def parse_args() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/config.yaml")
    return parser.parse_args()


def validate_config(config: Config) -> None:
    """Validate config arguments.

    Args:
    ----
        config: The main config.
    """
    preprocess_args = config.dataset_preprocess
    if preprocess_args.get("from_disk"):
        if "load_path" not in preprocess_args:
            msg = "`from_disk` set but `load_path` missing."
            raise KeyError(msg)
        path_exists = os.path.exists(preprocess_args.get("from_disk"))
        if not path_exists:
            msg = "`load_path` does not exist."
            raise Exception(msg)

    if preprocess_args.get("truncate") and preprocess_args.get("packing_type"):
        msg = "`truncate` and `packing_type` cannot both be set."
        raise ValueError(msg)

    if (
        not preprocess_args.get("truncate")
    ) and (
        not preprocess_args.get("packing_type")
    ):
        print(
            "Warning: neither `truncate` nor `packing_type` are set. This ",
            "can cause issues during the forward pass of the model if ",
            "tokenized example lengths exceed max sequence lengths.",
        )

    if not preprocess_args.get("save_path"):
        msg = "`save_path` missing from config."
        raise KeyError(msg)


def tokenize_dataset(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    data_field: str,
    pre_pend: str | None = None,
    truncate: bool = False,
    separator: str | None = None,
    add_bos_eos: bool = True,
) -> dict[str, list[int]]:
    """Tokenize the text dataset.

    Args:
    ----
        examples: The dictionary containing raw examples.
        tokenizer: The tokenizer.
        data_field: The dictionary key containing the examples to be tokenized.
        pre_pend: An optional prompt you can prepend.
        truncate: Whether to truncate examples that exceed context length.
        separator: A string separator s.t. everything before the separator
            is not going to be backpropped, and everything after will. Similar
            to supervised finetuning.
        add_bos_eos: Whether to add the BOS and EOS tokens to the tokenized
            sequence.

    Returns:
    -------
        A dictionary containing the input ids, labels, and attention masks.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    # Adding bos/eos
    if add_bos_eos:
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
    else:
        bos, eos = "", ""
    for example in examples[data_field]:
        # If we want to include a prepended prompt to each datapoint
        if pre_pend:
            prompt = f"{bos}{pre_pend}{example}{eos}"
        else:
            prompt = f"{bos}{example}{eos}"
        # If we've specified a separator present in each sequence
        if not separator:
            tokenized = tokenizer.encode(prompt, add_special_tokens=False)
            if truncate and len(tokenized) > tokenizer.model_max_length:
                tokenized = tokenized[:tokenizer.model_max_length - 1]
                tokenized.append(tokenizer.eos_token_id)
            all_labels.append(deepcopy(tokenized))
        else:
            if separator not in prompt:
                continue
            # Perform tokenization separately to allow for conditional prompting
            separation_idx = prompt.find(separator) + len(separator)
            prefix, postfix = prompt[:separation_idx], prompt[separation_idx:]
            tokenized_prefix = tokenizer.encode(
                prefix, add_special_tokens=False,
            )
            tokenized_postfix = tokenizer.encode(
                postfix, add_special_tokens=False,
            )
            tokenized = tokenized_prefix + tokenized_postfix
            if truncate and len(tokenized) > tokenizer.model_max_length:
                tokenized = tokenized[:tokenizer.model_max_length - 1]
                tokenized.append(tokenizer.eos_token_id)
            # We need to address this separately, because labels need to
            # backprop on bos/eos tokens
            if add_bos_eos:
                label = (
                    [tokenizer.bos_token_id]
                    + ([-100] * (len(tokenized_prefix) - 1))
                    + deepcopy(tokenized_postfix)
                )
            else:
                label = (
                    [-100] * len(tokenized_prefix)
                    + deepcopy(tokenized_postfix)
                )
            # If truncated, labels should be the same.
            if truncate and len(label) > tokenizer.model_max_length:
                label = label[:tokenizer.model_max_length - 1]
                label.append(tokenizer.eos_token_id)
            all_labels.append(label)
        all_input_ids.append(tokenized)
        all_attention_mask.append([1] * len(tokenized))

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def pack_examples(
    examples: dict[str, list[int]],
    tokenizer: PreTrainedTokenizer,
    overlap: int = 0,
    packing_type: str = "full",
    add_bos_eos: bool = True,
) -> dict[str, list[int]]:
    """Pack the tokenized dataset.

    Args:
    ----
        examples: The dictionary containing tokenized results.
        tokenizer: The tokenizer.
        overlap: The amount of overlap between two examples while packing.
        packing_type: In `partial` packing, original individual training
            examples are chunked with the option of `overlap`. In `full`
            packing, the entire dataset is fully packed meaning that no
            empty space is left in sequences.
        add_bos_eos: Whether to add the BOS and EOS tokens to the tokenized
            and packed sequence.

    Returns:
    -------
        Same type of input dictionary, but now with examples packed.
    """
    chunk_size = tokenizer.model_max_length
    if add_bos_eos:
        # For BOS and EOS tokens.
        chunk_size -= 2
        bos, eos = [tokenizer.bos_token_id], [tokenizer.eos_token_id]
    else:
        bos, eos = [], []
    stride = chunk_size - overlap
    all_keys = list(examples.keys())
    if packing_type == "full":
        joined_examples = {k: sum(examples[k], []) for k in all_keys}
        total_length = len(joined_examples["input_ids"])
        result = {}
        for k, v in joined_examples.items():
            value_chunked_lst = []
            for i in range(0, total_length, stride):
                if k != "attention_mask":
                    value_chunked_lst.append(bos + v[i:i + chunk_size] + eos)
                else:
                    if add_bos_eos:
                        # Need to do this explicitly because attention mask
                        # is just 1s or 0s.
                        value_chunked_lst.append(
                            [1] + v[i:i + chunk_size] + [1]
                        )
                    else:
                        value_chunked_lst.append(v[i:i + chunk_size])
    elif packing_type == "partial":
        result = {k:[] for k in examples}
        _key = all_keys[0]
        for idx in range(len(examples[_key])):
            total_length = len(examples[_key][idx])
            for key in all_keys:
                for i in range(0, total_length, stride):
                    if key != "attention_mask":
                        sliced_example = [
                            bos + examples[key][idx][i:i + chunk_size] + eos
                        ]
                    else:
                        if add_bos_eos:
                            sliced_example = [
                                [1] + examples[key][idx][i:i + chunk_size] + [1]
                            ]
                        else:
                            sliced_example = [
                                examples[key][idx][i:i + chunk_size]
                            ]
                    result[key].extend(sliced_example)
    else:
        msg = "`packing_type` needs to either be `full` or `partial`."
        raise ValueError(msg)
    return result


def add_indices(
    dataset: datasets.Dataset,
    base_idx: int = 0,
) -> datasets.Dataset:
    """Add the `id` column to examples. This is used for dataset checkpointing.

    Args:
    ----
        dataset: The dataset we are adding the column to.
        base_idx: An optional base index to start the ids from. Default is 0.

    Returns:
    -------
        The dataset with the new `id` column.
    """
    indices = [i + base_idx for i in range(len(dataset))]
    return dataset.add_column("id", indices)


def main(config: Config) -> None:
    """Definition of main function.

    Args:
    ----
        config: The main config.
    """
    preprocess_args = config.dataset_preprocess
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.model_max_length = config.train_parameters.max_seq_len
    if preprocess_args.from_disk:
        ds = datasets.load_from_disk(preprocess_args.load_path)
        if isinstance(ds, DatasetDict):
            if not preprocess_args.get("split"):
                msg = "Loaded dataset is a dictionary. Please specify `split`."
                raise KeyError(msg)
            ds = ds[preprocess_args.get("split")]
    else:
        ds = datasets.load_dataset(
            preprocess_args.load_path,
            split=preprocess_args.get("split"),
        )

    will_pack = False
    if preprocess_args.get("packing_type"):
        will_pack = True

    if preprocess_args.get("add_bos_eos_tokens", True):
        special_tokens_created = isinstance(
            tokenizer.bos_token_id, int,
        ) and isinstance(
            tokenizer.eos_token_id, int,
        )
        if not special_tokens_created:
            msg = (
                "BOS and EOS tokens are not set in the tokenizer.",
                "Cannot add these tokens during tokenization.",
            )
            raise TypeError(msg)

    ds = ds.map(
        lambda examples: tokenize_dataset(
            examples,
            tokenizer,
            preprocess_args.data_field,
            preprocess_args.get("pre_pend"),
            preprocess_args.get("truncate", False),
            preprocess_args.get("seperator"),
            # Note that if packing, we add the special tokens after packing.
            not will_pack and preprocess_args.get("add_bos_eos_tokens", True),
        ),
        batched=True,
        batch_size=250,
        remove_columns=ds.column_names,
        num_proc=32,
    )
    if preprocess_args.get("packing_type"):
        ds = ds.map(
            lambda examples: pack_examples(
                examples,
                tokenizer,
                preprocess_args.get("overlap", 0),
                preprocess_args.packing_type,
                will_pack and preprocess_args.get("add_bos_eos_tokens", True),
            ),
            batched=True,
            batch_size=2000,
            remove_columns=ds.column_names,
            num_proc=8,
        )
    ds = add_indices(ds)
    ds.save_to_disk(preprocess_args.save_path, max_shard_size="1GB")


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config_path)
    validate_config(config)
    main(config)
