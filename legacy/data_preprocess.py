from transformers import LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer
import json
from typing import List, Literal, Optional, Tuple, TypedDict, Union
import datasets
from functools import partial
import math
import os
import re
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from data_utils import Config, pack_examples, DataCollatorWithPadding


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

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


def llama_two_conversion(examples, tokenizer):
     all_labels = []
     all_input_ids = []
     all_attention_mask = []
     BOS, EOS = tokenizer.bos_token, tokenizer.eos_token
     for example in examples:
        # print(example)
        labels = []
        input_ids = []
        processed_convo = convert_to_qa(example)
        for single_qa in processed_convo:
            user, ai = single_qa
            if len(input_ids) == 0:
                tokenized_user = tokenizer.encode(f"{BOS}{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{user} {E_INST}", add_special_tokens=False)
            else:
                tokenized_user = tokenizer.encode(f"{BOS}{B_INST} {user} {E_INST}", add_special_tokens=False)
            tokenized_ai = tokenizer.encode(f" {ai} {EOS}", add_special_tokens=False)
            labels += [-100] * len(tokenized_user) + tokenized_ai
            input_ids += tokenized_user + tokenized_ai
        all_labels.append(labels)
        all_input_ids.append(input_ids)
        attention_mask = [1] * len(input_ids)
        all_attention_mask.append(attention_mask)
     return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}


def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs

# dialogs = read_dialogs_from_file("chats.json")

def add_indices(dataset, base_idx=0):
    indices = [i + base_idx for i in range(len(dataset))]
    dataset = dataset.add_column("id", indices)
    return dataset


def mdpi_mtb_dataset_wrap_pack(examples, max_length):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    new_indices = []
    overlap = 100
    stride = max_length - overlap
    for i, (
        seq, attention_mask, labels, id
    ) in enumerate(zip(
        examples["input_ids"],
        examples["attention_mask"],
        examples["labels"],
        examples["id"],
    )):
        if len(seq) > max_length:
            all_input_ids.append(seq[0: max_length])
            all_attention_mask.append(attention_mask[0: max_length])
            new_indices.append(deepcopy(id))
            all_labels.append(labels[0: max_length])

            begin_idx, end_idx = stride, stride + max_length

            while begin_idx < len(seq):
                curr_chunk_seq = seq[begin_idx: end_idx]
                curr_chunk_att = attention_mask[begin_idx: end_idx]
                curr_chunk_labels = labels[begin_idx: end_idx]
                curr_chunk_id = deepcopy(id)
                if begin_idx + stride >= len(seq) and len(curr_chunk_seq) < max_length:
                    # last iter and there's space to pack
                    if i < len(examples['id']) - 1:
                        # no packing for last example
                        space_remaining = max_length - len(curr_chunk_seq)

                        keys = ["input_ids", "attention_mask", "labels", "id"]
                        vars = [curr_chunk_seq, curr_chunk_att, curr_chunk_labels, curr_chunk_id]

                        for var, key in zip(vars, keys):
                            var.extend(examples[key][i + 1][:space_remaining])
                            if key != "id":
                                examples[key][i + 1] = examples[key][i + 1][space_remaining:]

                all_input_ids.append(curr_chunk_seq)
                all_attention_mask.append(curr_chunk_att)
                new_indices.append(curr_chunk_id)
                all_labels.append(curr_chunk_labels)
                begin_idx += stride
                end_idx += stride
        else:
            all_input_ids.append(seq)
            all_attention_mask.append([1] * len(seq))
            all_labels.append(deepcopy(seq))
            new_indices.append(id)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels, "id": new_indices}


def mdpi_mtb_dataset_tokenize(examples, tokenizer, indices_to_delete=[]):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    new_indices = []
    BOS, EOS = tokenizer.bos_token, tokenizer.eos_token
    for idx, example in zip(examples["id"], examples["text"]):
        if idx in indices_to_delete:
            continue
        prompt = f"{BOS}{example}{EOS}"
        tokenized = tokenizer.encode(prompt, add_special_tokens=False)
        all_input_ids.append(tokenized)
        all_attention_mask.append([1] * len(tokenized))
        all_labels.append(deepcopy(tokenized))
        new_indices.append([idx])
    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels, "id": new_indices}

def flatten_list_unique(ids: List[torch.Tensor]):
    curr_lst = []
    for item in ids:
        if len(item) == 1:
            curr_lst.append(item.item())
        else:
            curr_lst.extend(item.tolist())
    return torch.tensor(list(set(curr_lst)))

def get_highest_integer_folder(folder_path):
    folder_pattern = re.compile(r'^checkpoint_(\d+)$')

    max_integer = -1
    max_folder_name = None

    for folder_name in os.listdir(folder_path):
        match = folder_pattern.match(folder_name)
        if match:
            current_integer = int(match.group(1))
            if current_integer > max_integer:
                max_integer = current_integer
                max_folder_name = folder_name

    return max_folder_name

def remove_unwanted_rows(examples, rows):
    ids = examples['id']
    assertion_lst = []
    for id in ids:
        if id in rows:
            assertion_lst.append(False)
        else:
            assertion_lst.append(True)
    assert len(assertion_lst) == len(ids), f'Length of assertion list is {len(assertion_lst)}, expected {len(ids)}'
    return assertion_lst

def detokenize(examples, tokenizer):
    all_text = []
    for input_id in examples["input_ids"]:
        text = tokenizer.decode(input_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        all_text.append(text)
    return {"text": all_text}


if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained('/model-weights/Llama-2-7b-hf/')
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    config = Config(yaml_path='configs/config.yaml')
    tokenizer.model_max_length = config.train.max_seq_len
    train_set = datasets.load_from_disk("/checkpoint/opt_test/original/clinical_llm/datasets/mdpi_mtb_processed/train/")
    test_set = datasets.load_from_disk("/checkpoint/opt_test/original/clinical_llm/datasets/mdpi_mtb_processed/test/")
    test_set = test_set.map(
        lambda examples: detokenize(examples, tokenizer),
        batched=True,
        batch_size=2000,
        remove_columns=test_set.column_names,
        num_proc=16,
    )
    train_set = train_set.map(
        lambda examples: detokenize(examples, tokenizer),
        batched=True,
        batch_size=2000,
        remove_columns=train_set.column_names,
        num_proc=16,
    )
    train_set.save_to_disk(
        f"/checkpoint/opt_test/original/clinical_llm/datasets/mdpi_mtb_processed_{config.train.max_seq_len}_detokenized/train"
    )
    test_set.save_to_disk(
        f"/checkpoint/opt_test/original/clinical_llm/datasets/mdpi_mtb_processed_{config.train.max_seq_len}_detokenized/test"
    )
    quit()

    mtb_dataset = datasets.load_from_disk("/checkpoint/opt_test/original/clinical_llm/datasets/mtb")['train'] # store shuffled ds on disk for determinism
    mtb_dataset = mtb_dataset.rename_column("texts", "text")
    mtb_dataset = mtb_dataset.train_test_split(test_size=0.248, shuffle=False)
    mtb_dataset_train = mtb_dataset['train']
    mtb_dataset_train = mtb_dataset_train.sort('index')
    mtb_dataset_test = mtb_dataset['test']
    mtb_dataset_train = mtb_dataset_train.remove_columns(["textbook", "index"])
    mtb_dataset_test = mtb_dataset_test.remove_columns(["textbook", "index"])

    mdpi_dataset = datasets.load_from_disk("/checkpoint/opt_test/original/clinical_llm/datasets/mdpi")
    mdpi_dataset_test = mdpi_dataset['test']
    mdpi_dataset_leftover_test_split = mdpi_dataset_test.train_test_split(test_size=0.0424, shuffle=False)
    mdpi_dataset_test = mdpi_dataset_leftover_test_split['test']
    mdpi_dataset_train = datasets.concatenate_datasets([
        mdpi_dataset['train'],
        mdpi_dataset['validation'],
        mdpi_dataset_leftover_test_split['train'],
    ])


    # only save processed indices of train set during checkpointing
    # eval uses whole test set, so only adding indices for uniformity of code
    mtb_dataset_train = add_indices(mtb_dataset_train)
    mdpi_dataset_train = add_indices(mdpi_dataset_train, len(mtb_dataset_train))
    mtb_dataset_test = add_indices(mtb_dataset_test)
    mdpi_dataset_test = add_indices(mdpi_dataset_test, len(mtb_dataset_test))

    mdpi_dataset_train = mdpi_dataset_train.shuffle() # don't shuffle mtb dataset (continual learning)
    train_dataset = datasets.concatenate_datasets([mtb_dataset_train, mdpi_dataset_train])
    test_dataset = datasets.concatenate_datasets([mtb_dataset_test, mdpi_dataset_test])

    test_dataset = test_dataset.map(
        lambda examples: mdpi_mtb_dataset_tokenize(examples, tokenizer),
        batched=True,
        batch_size=2000,
        remove_columns=["text"],
        num_proc=16,
    )
    test_dataset = test_dataset.map(
        lambda examples: mdpi_mtb_dataset_wrap_pack(examples, config.train.max_seq_len),
        batched=True,
        batch_size=2000,
        num_proc=16,
        remove_columns=test_dataset.column_names
    )
    test_dataset = test_dataset.rename_column("id", "raw_data_id")
    test_dataset = test_dataset.add_column('id', [i for i in range(len(test_dataset))])
    test_dataset.save_to_disk(f"/checkpoint/opt_test/original/clinical_llm/datasets/mdpi_mtb_processed_{config.train.max_seq_len}/test")

    train_dataset = train_dataset.map(
        lambda examples: mdpi_mtb_dataset_tokenize(examples, tokenizer),
        batched=True,
        batch_size=5000,
        remove_columns=["text"],
        num_proc=32,
    )
    train_dataset = train_dataset.map(
        lambda examples: mdpi_mtb_dataset_wrap_pack(examples, config.train.max_seq_len),
        batched=True,
        batch_size=2000,
        num_proc=16,
        remove_columns=train_dataset.column_names
    )
    train_dataset = train_dataset.shuffle()  # TODO: remove for continual learning
    train_dataset = train_dataset.rename_column("id", "raw_data_id")
    train_dataset = train_dataset.add_column('id', [i for i in range(len(train_dataset))])
    train_dataset.save_to_disk(f"/checkpoint/opt_test/original/clinical_llm/datasets/mdpi_mtb_processed_{config.train.max_seq_len}/train")
