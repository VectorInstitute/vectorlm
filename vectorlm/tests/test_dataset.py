"""Unit tests for the in-memory benchmarking dataset."""

import pytest
from box import Box
from transformers import AutoTokenizer

from profiling.benchmark import BenchmarkingDataset
from vectorlm.tests.test_modelling import (
    _setup_and_teardown_torch_process_group,
)

_BATCH_TOKEN_DIMENSIONALITY = 2

dataset_config = Box(
    {
        "ignore_index": -100,
        "eval_bs": 8,
        "train_bs": 8,
        "train_ds": "/dev/null",
        "eval_ds": "/dev/null",
    },
)


@pytest.fixture()
def benchmark_dataset(
    _setup_and_teardown_torch_process_group, # noqa: ANN001
) -> BenchmarkingDataset:
    """Instantiate example in-memory benchmarking dataset."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    return BenchmarkingDataset(
        config=dataset_config,
        num_train_examples=10000,
        num_eval_examples=1000,
        tokenizer=tokenizer,
    )


def test_initialize_dataset(benchmark_dataset: BenchmarkingDataset) -> None:
    """Ensure that instantiating dataset does not throw an error message."""
    print(benchmark_dataset)


def test_get_batch(benchmark_dataset: BenchmarkingDataset) -> None:
    """Verify shape of dataset iterator output."""
    benchmark_dataset.setup_dataloaders()
    dataset_iterator = iter(benchmark_dataset.train_dataloader)
    batch = next(dataset_iterator)

    for key in ["input_ids", "attention_mask"]:
        assert (
            len(batch[key].shape) == _BATCH_TOKEN_DIMENSIONALITY
        )  # batch, tokens

    print(batch)
