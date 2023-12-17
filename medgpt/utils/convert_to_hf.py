import argparse
import os
from argparse import Namespace

import torch

from medgpt.utils.data_utils import Config
from medgpt.utils.model_utils import load_model_and_tokenizer


def parse_args() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="../configs/config.yaml")
    return parser.parse_args()

def converter(config: Config) -> None:
    """Define main converting function.

    Args:
    ----
        config: The full config.
    """
    state_dict = torch.load(
        os.path.join(
            config.train_parameters.output_dir,
            "final-model",
            "model.bin",
        ),
    )
    model, _ = load_model_and_tokenizer(
        config.model,
        True,
        False,
        2048,
    )
    model.load_state_dict(state_dict)
    model.save_pretrained(
        os.path.join(config.train_parameters.output_dir, "hf-model"),
    )

if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config_path)
    converter(config)
