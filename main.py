from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    pipeline,
    set_seed,
    get_scheduler,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import wandb
import datasets
import argparse
import math
import os
from tqdm import tqdm
import sys
from typing import List
import re
from copy import deepcopy
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import torch.distributed

from opengpt.config import Config
from opengpt.model_utils import add_tokens_to_model_and_tokenizer
from opengpt.dataset_utils import create_labels, pack_examples
from opengpt.data_collator import DataCollatorWithPadding


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--yaml_path", default="../configs/example_train_config.yaml", required=False
    )
    return parser.parse_args()


def setup_accelerator(cfg, training_args):
    accelerator = Accelerator(
        log_with="wandb",
    )
    print(f"Process {accelerator.process_index} started")
    if accelerator.is_main_process:
        os.makedirs(
            os.path.join(training_args.output_dir, "final-model"), exist_ok=True
        )

    wandb_dict = cfg.train.wandb_config.to_dict()
    accelerator.init_trackers(
        project_name=wandb_dict.pop("project_name"),
        config=cfg.train.hf_training_arguments.to_dict(),
        init_kwargs=wandb_dict,
    )
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    if accelerator.is_main_process:
        wandb.define_metric("millions_of_tokens")
        wandb.define_metric("train/*", step_metric="millions_of_tokens")
        wandb.define_metric("test/*", step_metric="millions_of_tokens")
    return accelerator, wandb_tracker


def load_model_and_tokenizer(cfg):
    model = LlamaForCausalLM.from_pretrained(cfg.train.model)
    
    tokenizer = LlamaTokenizer.from_pretrained(cfg.train.model)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.model_max_length = cfg.train.max_seq_len
    return model, tokenizer


def save_metadata(
    out_dir: str,
    accelerator: Accelerator,
    meta_dict,
    tr_step,
    epoch,
):
    save_dir = os.path.join(
        out_dir,
        "checkpoints",
        f"epoch_{epoch}",
        f"checkpoint_{tr_step}",
    )
    os.makedirs(save_dir, exist_ok=True)
    accelerator.save(meta_dict, os.path.join(save_dir, "meta_data.pkl"))

    accelerator.wait_for_everyone()


def load_metadata(
    in_dir: str,
    accelerator: Accelerator,
):
    latest_checkpoint_folder = get_latest_checkpoint_dir(
        os.path.join(in_dir, "checkpoints")
    )
    save_dir = os.path.join(
        in_dir,
        "checkpoints",
        latest_checkpoint_folder,
    )
    accelerator.print(f"Checkpoint found at {save_dir}")
    save_path = os.path.join(save_dir, "meta_data.pkl")
    meta_dict = torch.load(save_path)
    checkpointed_step = meta_dict["tr_step"]
    checkpointed_epoch = meta_dict["epoch"]
    to_remove = meta_dict["processed_ids"].int().tolist()
    return checkpointed_step, checkpointed_epoch, to_remove


def get_latest_checkpoint_dir(folder_path):
    epoch_pattern = re.compile(r"^epoch_(\d+)$")
    folder_pattern = re.compile(r"^checkpoint_(\d+)$")

    def find_largest(pattern, folder):
        max_integer = -1
        max_folder_name = None

        for folder_name in os.listdir(folder):
            match = pattern.match(folder_name)
            if match:
                current_integer = int(match.group(1))
                if current_integer > max_integer:
                    max_integer = current_integer
                    max_folder_name = folder_name
        return max_folder_name

    epoch_folder = find_largest(epoch_pattern, folder_path)
    folder_path = os.path.join(folder_path, epoch_folder)
    checkpoint_folder = find_largest(folder_pattern, folder_path)
    return os.path.join(epoch_folder, checkpoint_folder)


def checkpoint_exists(output_dir: str):
    if os.path.isdir(os.path.join(output_dir, "checkpoints")):
        return True
    return False


def llama_two_nhs_conversion(examples, tokenizer):

    def convert_to_qa(example):
        processed_full_convo = []
        full_convo = [i.strip() for i in example.split("<|user|>")[1:]]
        for one_pass in full_convo:
            if "<|ai|>" not in one_pass:
                continue
            user, ai = one_pass.split("<|ai|>")
            user = user.replace("<|eos|>", "").replace("<|eod|>", "").strip()
            ai = ai.replace("<|eos|>", "").replace("<|eod|>", "").strip()
            processed_full_convo.append([user, ai])
        return processed_full_convo

    all_labels = []
    all_input_ids = []
    all_attention_mask = []
    BOS, EOS = tokenizer.bos_token, tokenizer.eos_token
    for example in examples:
        labels = []
        input_ids = []
        processed_convo = convert_to_qa(example)
        for single_qa in processed_convo:
            user, ai = single_qa
            if len(input_ids) == 0:
                tokenized_user = tokenizer.encode(
                    f"{BOS}{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{user} {E_INST}",
                    add_special_tokens=False,
                )
            else:
                tokenized_user = tokenizer.encode(
                    f"{BOS}{B_INST} {user} {E_INST}", add_special_tokens=False
                )
            tokenized_ai = tokenizer.encode(f" {ai} {EOS}", add_special_tokens=False)
            labels += [-100] * len(tokenized_user) + tokenized_ai
            input_ids += tokenized_user + tokenized_ai
        all_labels.append(labels)
        all_input_ids.append(input_ids)
        attention_mask = [1] * len(input_ids)
        all_attention_mask.append(attention_mask)
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def remove_unwanted_rows(examples, rows):
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


def get_mdpi_mtb_dataloaders(
    accelerator: Accelerator,
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    config: Config,
    to_remove: List[int],
    checkpoint: bool,
):
    """
    Returns the MDPI/MTB dataset train/test dataloaders
    """
    # batch sizes
    train_bs = config.train.hf_training_arguments.per_device_train_batch_size
    eval_bs = config.train.hf_training_arguments.per_device_eval_batch_size

    if not checkpoint:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings[-1:] = input_embeddings_avg
        output_embeddings[-1:] = output_embeddings_avg
    else:
        # simply just resize w/ random weights and load ckpt later
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # load dataset
    train_dataset = datasets.load_from_disk(
        "/checkpoint/opt_test/original/clinical_llm/llama-2-7b-ft-mdpi-mtb/datasets/mdpi_mtb_processed/train"
    )
    test_dataset = datasets.load_from_disk(
        "/checkpoint/opt_test/original/clinical_llm/llama-2-7b-ft-mdpi-mtb/datasets/mdpi_mtb_processed/test"
    )

    orig_length = math.ceil(len(train_dataset) / train_bs)

    # filter out unwanted rows from data we've already trained on
    if to_remove != []:
        to_remove = set(to_remove)

        train_dataset = train_dataset.filter(
            lambda example: remove_unwanted_rows(example, to_remove),
            batched=True,
            num_proc=8,
            batch_size=5000,
        )

    accelerator.print(f"Train dataset length {len(train_dataset)} (after wrapping)")
    accelerator.print(f"Eval dataset length {len(test_dataset)} (after wrapping)")

    dc = DataCollatorWithPadding(
        tokenizer.pad_token_id,
        config.train.ignore_index,
        max_seq_len=config.train.max_seq_len,
    )
    eval_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=dc, batch_size=eval_bs
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=dc, batch_size=train_bs
    )
    return orig_length, train_dataloader, eval_dataloader


def reset_mdpi_mtb_dataloader(
    tokenizer: LlamaTokenizer,
    cfg: Config,
):
    train_bs = cfg.train.hf_training_arguments.per_device_train_batch_size
    train_dataset = datasets.load_from_disk(
        "/checkpoint/opt_test/original/clinical_llm/llama-2-7b-ft-mdpi-mtb/datasets/mdpi_mtb_processed/train"
    )

    dc = DataCollatorWithPadding(
        tokenizer.pad_token_id,
        cfg.train.ignore_index,
        max_seq_len=cfg.train.max_seq_len,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=dc, batch_size=train_bs
    )

    return train_dataloader


def get_nhs_dataloaders(
    accelerator: Accelerator,
    model: LlamaForCausalLM,
    config: Config,
    checkpoint: bool,
    to_remove: List[int],
):
    """
    Returns the NHS dataset train/test dataloaders
    """
    # batch sizes
    train_bs = config.train.hf_training_arguments.per_device_train_batch_size
    eval_bs = config.train.hf_training_arguments.per_device_eval_batch_size

    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        "/ssd005/projects/llm/Llama-2-7b-chat-hf/"
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.model_max_length = config.train.max_seq_len

    if not checkpoint:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings[-1:] = input_embeddings_avg
        output_embeddings[-1:] = output_embeddings_avg
    model.config.pad_token_id = tokenizer.pad_token_id

    # load dataset
    dataset = datasets.Dataset.from_csv(config.train.datasets)
    accelerator.wait_for_everyone()
    to_remove_columns = list(dataset.column_names)
    to_remove_columns.remove("text")
    dataset = dataset.remove_columns(to_remove_columns)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # data collator with padding
    dc = DataCollatorWithPadding(
        tokenizer.pad_token_id,
        config.train.ignore_index,
        max_seq_len=config.train.max_seq_len,
    )

    train_dataset = train_dataset.map(
        lambda example: llama_two_nhs_conversion(example["text"], tokenizer),
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )

    accelerator.wait_for_everyone()

    test_dataset = test_dataset.map(
        lambda example: llama_two_nhs_conversion(example["text"], tokenizer),
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )
    test_dataset = test_dataset.add_column("id", [i for i in range(len(test_dataset))])

    accelerator.wait_for_everyone()

    # We only do packing for the train set
    train_dataset = train_dataset.map(
        lambda examples: pack_examples(
            examples, config.train.max_seq_len, packing_type=config.train.packing_type
        ),
        batched=True,
        batch_size=1000,
        num_proc=1,
    )
    train_dataset = train_dataset.add_column(
        "id", [i for i in range(len(train_dataset))]
    )
    orig_length = math.ceil(len(train_dataset) / train_bs)

    accelerator.wait_for_everyone()

    if to_remove != []:
        to_remove = set(to_remove)

        train_dataset = train_dataset.filter(
            lambda example: remove_unwanted_rows(example, to_remove),
            batched=True,
            num_proc=8,
            batch_size=5000,
        )

    accelerator.print(f"Train dataset length {len(train_dataset)} (after packing)")
    accelerator.print(f"Eval dataset length {len(test_dataset)} (no packing)")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=dc, batch_size=train_bs
    )

    eval_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=dc, batch_size=eval_bs
    )

    return orig_length, train_dataloader, eval_dataloader


def train_function(cfg):
    training_args = cfg.train.hf_training_arguments
    checkpoint = checkpoint_exists(training_args.output_dir)
    checkpointed_step = 0
    checkpointed_epoch = 0
    set_seed(training_args.seed)

    accelerator, wandb_tracker = setup_accelerator(cfg, training_args)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print(f"Rank: {rank}, World size: {world_size}")
    if checkpoint:
        accelerator.print("Checkpoint found")

    to_remove = []
    model, tokenizer = load_model_and_tokenizer(cfg)
    if checkpoint:
        checkpointed_step, checkpointed_epoch, to_remove = load_metadata(
            training_args.output_dir,
            accelerator,
        )

    accelerator.wait_for_everyone()
    orig_length, train_dataloader, eval_dataloader = get_mdpi_mtb_dataloaders(
        accelerator,
        model,
        tokenizer,
        cfg,
        to_remove,
        checkpoint,
    )
    # orig_length, train_dataloader, eval_dataloader = get_nhs_dataloaders(
    #     accelerator,
    #     model,
    #     cfg,
    #     checkpoint,
    #     to_remove
    # )

    model = accelerator.prepare(model)
    accelerator.wait_for_everyone()

    accelerator.print("Dataloading complete")
    accelerator.print(
        f"Train/Eval dataloader sizes before sharding are {len(train_dataloader)}, {len(eval_dataloader)}"
    )

    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    orig_length = math.ceil(orig_length / world_size)
    # num_update_steps_per_epoch = max(
    #     len(train_dataloader) // gradient_accumulation_steps, 1
    # )
    num_update_steps_per_epoch = max(
        orig_length // gradient_accumulation_steps, 1
    )

    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    logging_steps = training_args.logging_steps
    saving_steps = int(training_args.save_frequency * orig_length)

    optimizer = AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
    )

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        math.ceil(max_steps * training_args.warmup_ratio * world_size),
        max_steps * world_size,
    )
    # num warm up steps is times world_size because each .step() call
    # increments the internal step count by <num_processes>

    accelerator.print("Optimizers and LR scheduler created")
    accelerator.print(
        f"Original dataloader length: {orig_length * world_size}, "
        f"per device *update* steps: {max_steps}, warmup steps "
        f"{math.ceil(max_steps * training_args.warmup_ratio)}, " 
        f"training for {training_args.num_train_epochs} epochs "
        f"which is a total of "
        f"{orig_length * training_args.num_train_epochs} "
        f"per device iterations"
    )

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    if checkpoint:
        latest_checkpoint_folder = get_latest_checkpoint_dir(
            os.path.join(training_args.output_dir, "checkpoints")
        )
        save_dir = os.path.join(
            training_args.output_dir,
            "checkpoints",
            latest_checkpoint_folder,
        )
        accelerator.load_state(save_dir)
        accelerator.print("All states restored")

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

    accelerator.print("Sharded")
    accelerator.print(
        f"Train/Eval dataloader sizes after sharding are {len(train_dataloader)}, {len(eval_dataloader)}"
    )

    accelerator.wait_for_everyone()

    for epoch in range(checkpointed_epoch, training_args.num_train_epochs):
        model.train()

        if checkpoint:
            processed_ids = torch.tensor(to_remove).to(accelerator.device)
        else:
            processed_ids = torch.tensor([]).to(accelerator.device)

        train_dl_iterator = iter(train_dataloader)

        for step in tqdm(
            range(len(train_dataloader)),
            disable=not accelerator.is_main_process,
            file=sys.__stdout__,
        ):
            tr_step = checkpointed_step + step
            batch = next(train_dl_iterator)

            # saving
            if (
                (tr_step + 1) % saving_steps == 0
            ) and training_args.checkpointing_enabled:
                gathered_processed_ids = accelerator.gather(processed_ids)
                meta_dict = {
                    "tr_step": tr_step + 1,
                    "processed_ids": gathered_processed_ids,
                    "epoch": epoch,
                }
                save_dir = os.path.join(
                    training_args.output_dir,
                    "checkpoints",
                    f"epoch_{epoch}",
                    f"checkpoint_{tr_step}",
                )
                accelerator.save_state(save_dir)
                save_metadata(
                    training_args.output_dir, accelerator, meta_dict, tr_step, epoch
                )

            # training
            ids = batch.pop("id")
            batch.pop("raw_data_id") if "raw_data_id" in batch else None
            processed_ids = torch.cat([processed_ids, ids])
            if (
                tr_step + 1
            ) % gradient_accumulation_steps != gradient_accumulation_steps - 1:
                # no need to sync while accumulating gradients
                with accelerator.no_sync(model):
                    out = model(**batch)
                    tr_step_loss = out.loss
                    accelerator.backward(tr_step_loss / gradient_accumulation_steps)
                    accelerator.clip_grad_norm_(
                        model.parameters(), training_args.max_grad_norm
                    )
            else:
                # next forward / backward pass will be synced
                accelerator.wait_for_everyone()
                out = model(**batch)
                tr_step_loss = out.loss
                accelerator.backward(tr_step_loss / gradient_accumulation_steps)
                accelerator.clip_grad_norm_(
                    model.parameters(), training_args.max_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                accelerator.print(f"LR: {lr_scheduler.get_last_lr()[0]}")
                optimizer.zero_grad()
            gathered_tr_step_loss = accelerator.gather(tr_step_loss).mean().item()

            # logging
            if accelerator.is_main_process:
                num_tokens_processed = (
                    world_size
                    * (tr_step + 1 + orig_length * epoch)
                    * training_args.per_device_train_batch_size
                    * cfg.train.max_seq_len
                ) / 1e6
                commit = True if tr_step % logging_steps != 0 else False
                wandb_tracker.log(
                    {
                        "train/step_loss": gathered_tr_step_loss,
                        "millions_of_tokens": num_tokens_processed,
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    commit=commit,
                    step=tr_step + (epoch * orig_length),
                )

            # evaluating
            if tr_step % logging_steps == 0:
                accelerator.print("Evaluating")
                model.eval()
                eval_loss = torch.tensor(0.0).to(accelerator.device)
                for _, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        batch.pop("id")
                        batch.pop("raw_data_id") if "raw_data_id" in batch else None
                        out = model(**batch)
                        eval_loss += out.loss
                gathered_eval_loss = accelerator.gather(eval_loss).mean().item()
                if accelerator.is_main_process:
                    accelerator.print(
                        f"Step: {tr_step}, train loss: {gathered_tr_step_loss}, eval loss: {gathered_eval_loss / len(eval_dataloader)}"
                    )
                    wandb_tracker.log(
                        {
                            "test/loss": gathered_eval_loss / len(eval_dataloader),
                            "millions_of_tokens": num_tokens_processed,
                        },
                        step=tr_step + (epoch * orig_length),
                    )
                model.train()

        if checkpoint:
            checkpointed_step = 0
            checkpoint = False
        
        if training_args.num_train_epochs > 1:
            # reset dataset (add back processed ids)
            train_dataloader = reset_mdpi_mtb_dataloader(tokenizer, cfg)
            train_dataloader = accelerator.prepare(train_dataloader)

    accelerator.wait_for_everyone()

    # save model

    save_dir = os.path.join(
        training_args.output_dir,
        "final-model",
    )
    accelerator.save_state(save_dir)

    accelerator.wait_for_everyone()
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(training_args.output_dir, "final-model"),
        in_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        max_shard_size="10GB",
        state_dict=accelerator.get_state_dict(model),
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    config = Config(yaml_path=args.yaml_path)
    config.train.hf_training_arguments["output_dir"] = args.output_dir
    train_function(config)


if __name__ == "__main__":
    main()
