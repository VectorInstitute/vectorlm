from utils.data_utils import Config, Dataset
from utils.trainer import Trainer
from utils.optimizer_utils import get_custom_scheduler
from utils.save_utils import save_consolidated_model
from utils.misc_utils import setup, cleanup, wandb_setup
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import argparse
import os
import wandb
import torch.distributed as dist
from argparse import Namespace
from transformers import set_seed
import math
import torch
from torch.optim import AdamW
from tqdm import tqdm
import sys


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--yaml_path", default="configs/config.yaml", required=False
    )
    parser.add_argument("--ckpt_dir", default= "", required=False)  # TODO: remove
    return parser.parse_args()


def main(config: Config) -> None:
    training_args = config.train_parameters

    # set a seed
    set_seed(training_args.seed)

    # set CUDA related dependencies
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Rank: {rank}, World size: {world_size}")
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()

    # setup wandb
    if rank == 0:
        wandb_setup(training_args, **config.wandb_config)
    dist.barrier()

    # instantiate primitive classes
    trainer = Trainer(
        training_args,
        config.model,
        True,
    )
    trainer.shard_model(LlamaDecoderLayer) # full FSDP by default

    optimizer = AdamW(
        trainer.model.parameters(),
        **training_args.optimizer,
    )

    lr_scheduler = get_custom_scheduler(
        "cosine",
        optimizer,
        math.ceil(
            trainer.num_update_steps_per_epoch * training_args.warmup_ratio
        ),
        trainer.max_steps,
    )

    dataset = Dataset(
        config.dataset,
        trainer.tokenizer
    )

    trainer.set_optim_and_scheduler(optimizer, lr_scheduler)
    trainer.set_dataset(dataset)

    checkpointed_epoch = 0 # TODO: here for testing, remove it.
    checkpointed_step = 0

    for epoch in range(checkpointed_epoch, training_args.num_train_epochs):
        trainer.model.train()
        train_dl_iterator = iter(dataset.train_dataloader)
        for step in tqdm(
            range((dataset.train_dataloader)),
            disable=rank != 0,
            file=sys.__stdout__,
        ):
            tr_step = checkpointed_step + step
            batch = next(train_dl_iterator)
            trainer.train_step(batch, epoch)
            if tr_step % trainer.logging_steps == 0:
                trainer.eval_step(epoch)
    
    if epoch == training_args.num_train_epochs - 1:
        hf_save_dir = os.path.join(training_args.output_dir, "final-model")
    else:
        hf_save_dir = os.path.join(
            training_args.output_dir,
            "checkpoints",
            f"epoch_{epoch}",
            "hf_model",
        )
    save_consolidated_model(trainer.model, hf_save_dir, rank)

if __name__ == "__main__":
    args = parse_args()
    config = Config(yaml_path=args.yaml_path)
    config.train_parameters["output_dir"] = args.output_dir
    config.train_parameters["ckpt_dir"] = args.ckpt_dir
    setup(args.output_dir)
    main(config)
    cleanup()