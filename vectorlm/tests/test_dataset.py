import pytest
from vectorlm.tests.test_modelling import setup_and_teardown_torch_process_group
from profiling.benchmark import BenchmarkingDataset

from box import Box

from transformers import AutoTokenizer


dataset_config = Box(
    {
        "ignore_index": -100,
        "eval_bs": 8,
        "train_bs": 8,
        "train_ds": "/dev/null",
        "eval_ds": "/dev/null",
    }
)


@pytest.fixture()
def benchmark_dataset(setup_and_teardown_torch_process_group):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    return BenchmarkingDataset(dataset_config, tokenizer)  # type: ignore


def test_initialize_dataset(benchmark_dataset):
    print(benchmark_dataset)


def test_get_batch(benchmark_dataset):
    benchmark_dataset.setup_dataloaders()
    dataset_iterator = iter(benchmark_dataset.train_dataloader)
    batch = next(dataset_iterator)

    for key in ["input_ids", "attention_mask"]:
        assert len(batch[key].shape) == 2  # batch, tokens

    print(batch)
