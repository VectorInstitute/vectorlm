import torch
from box import Box
import jsonpickle
import os
import yaml
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.distributed as dist
from typing import List
import datasets
import math
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import random

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Your answers should not "
    "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
    "illegal content. Please ensure that your responses are socially "
    "unbiased and positive in nature.\nIf a question does not make any "
    "sense, or is not factually coherent, explain why instead of "
    "answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)

class BaseConfig(object):
    def __init__(self, to_box=False):
        pass

    def _to_box(self):
        # Convert all dicts to boxes
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                self.__setattr__(key, Box(val))

    def _from_box(self):
        # Convert all dicts to boxes
        for key, val in self.__dict__.items():
            if isinstance(val, Box):
                self.__setattr__(key, val.to_dict())

    def save(self, save_path=None):
        r''' Save the config into a .json file
        Args:
            save_path (`str`):
                Where to save the created json file, if nothing we use the default from paths.
        '''
        if save_path is None:
            save_path = self.path.self

        # We want to save the dict here, not the whole class
        self._from_box()
        json_string = jsonpickle.encode({k:v for k,v in self.__dict__.items() if k != 'path'})

        with open(save_path, 'w') as f:
            f.write(json_string)
        self._to_box()

    @classmethod
    def load(cls, save_path):
        config = cls(to_box=False)
        # Read the jsonpickle string
        with open(save_path) as f:
            config_dict = jsonpickle.decode(f.read())
        config.merge_config(config_dict)
        config._to_box()
        return config

    def merge_config(self, config_dict):
        r''' Merge a config_dict with the existing config object.
        Args:
            config_dict (`dict`):
                A dictionary which key/values should be added to this class.
        '''
        for key in config_dict.keys():
            if key in self.__dict__ and isinstance(self.__dict__[key], dict):
                self.__dict__[key].update(config_dict[key])
            else:
                self.__dict__[key] = config_dict[key]


class Config(BaseConfig):
    r''' There are probably nicer ways to do this, but I like this one.
    '''
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.load_yaml(yaml_path)

    def reload_yaml(self):
        self.load_yaml(self.yaml_path)

    def load_yaml(self, yaml_path):
        _config = yaml.safe_load(open(yaml_path, 'r'))
        self.to_box = True
        self.base_path = './'
        self.datasets = {}
        self.name = 'opengpt'

        for k,v in _config.items():
            self.__setattr__(k, v)
        # For fun, we will also keept the _config
        self._config = _config

        self.path = {'self': os.path.join(self.base_path, f'config_for_{self.name}.json')}
        if _config.get('static_paths', None):
            self.path.update(_config['static_paths'])

        if self.to_box:
            self._to_box()

            def create_dirs(paths):
                for path in paths:
                    if isinstance(path, str):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                    elif isinstance(path, dict):
                        create_dirs(path.values())
            create_dirs(self.path.values())
        
        # Create dirs for datasets, this is where all the data from one dataset will go
        for ds in self.datasets:
            os.makedirs(os.path.join(self.base_path, ds['name']), exist_ok=True)

class DataCollatorWithPadding(object):
    r''' Will pad or trim examples to the appropriate length.
    '''
    def __init__(self, pad_token_id, ignore_index, max_seq_len):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len

    def __call__(self, instances):
        batch = {}
        if 'id' in instances[0]:
            if 'raw_data_id' in instances[0]:
                keys = ["input_ids", "labels", "raw_data_id"]
                input_ids, labels, raw_data_id = tuple([torch.tensor(instance[key][0:self.max_seq_len]) for instance in instances] for key in keys)
                batch['raw_data_id'] = raw_data_id
            else:
                keys = ["input_ids", "labels"]
                input_ids, labels = tuple([torch.tensor(instance[key][0:self.max_seq_len]) for instance in instances] for key in keys)
            batch['id'] = torch.tensor([instance['id'] for instance in instances])
        else:
            keys = ["input_ids", "labels"]
            input_ids, labels = tuple([torch.tensor(instance[key][0:self.max_seq_len]) for instance in instances] for key in keys)
        
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id) 
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        batch['attention_mask'] = batch['input_ids'].ne(self.pad_token_id)
    
        return batch


def pack_examples(examples, block_size, packing_type='partial'):
    r''' Used with a prepared HF dataset, will pack/group examples. Use with care, can mess up many things
    if the input is not formated properly (requires the <|eod|> token).
    
    packing_type: partial/full/no 
    '''
    # Concatenate all texts.
    if packing_type == 'partial':
        result = {k:[] for k in examples.keys()}
        # _key = list(examples.keys())[0] # Take whichever key
        _key = 'input_ids'
        new_example = {k:[] for k in examples.keys()}

        for ind in range(len(examples[_key])):
            # Trim long sequences to block_size, this is required for partial packing
            example = {k:v[ind][0:block_size] for k,v in examples.items()}
            if len(new_example[_key]) + len(example[_key]) > block_size:
                result = {k:result[k] + [v] for k,v in new_example.items()}
                new_example = example 
            else:
                new_example = {k:new_example[k] + v for k,v in example.items()}
        #  Add the last example if there is something to add  
        if len(new_example[_key]) > 0:   
            result = {k:result[k] + [v] for k,v in new_example.items()}
    elif packing_type == 'full':
        # Full packing
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    else:
        # Do nothing
        result = examples
    return result

# TODO: fix below all
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
    tokenizer: LlamaTokenizer,
    config: Config,
    to_remove: List[int],
):
    """
    Returns the MDPI/MTB dataset train/test dataloaders
    """
    # batch sizes
    train_bs = config.train.hf_training_arguments.per_device_train_batch_size
    eval_bs = config.train.hf_training_arguments.per_device_eval_batch_size

    # load dataset
    train_dataset = datasets.load_from_disk(
        config.train.train_ds
    )
    test_dataset = datasets.load_from_disk(
       config.train.test_ds
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
    if dist.get_rank() == 0:
        print(f"Train dataset length {len(train_dataset)} (after wrapping)")
        print(f"Eval dataset length {len(test_dataset)} (after wrapping)")

    dc = DataCollatorWithPadding(
        tokenizer.pad_token_id,
        config.train.ignore_index,
        max_seq_len=config.train.max_seq_len,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        dist.get_world_size(),
        dist.get_rank(),
        # seed=random.randint(0, 30),
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        test_dataset, collate_fn=dc, batch_size=eval_bs, sampler=test_sampler, shuffle=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=dc,
        batch_size=train_bs,
        sampler=train_sampler,
        shuffle=False,
    )
    return orig_length, train_dataloader, eval_dataloader


def reset_mdpi_mtb_dataloader(
    tokenizer: LlamaTokenizer,
    cfg: Config,
):
    train_bs = cfg.train.hf_training_arguments.per_device_train_batch_size
    train_dataset = datasets.load_from_disk(
       cfg.train.train_ds
    )

    dc = DataCollatorWithPadding(
        tokenizer.pad_token_id,
        cfg.train.ignore_index,
        max_seq_len=cfg.train.max_seq_len,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        dist.get_world_size(),
        dist.get_rank(),
        seed=random.randint(0, 30),
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=dc,
        batch_size=train_bs,
        sampler=train_sampler,
    )

    return train_dataloader


def get_nhs_dataloaders(
    model: LlamaForCausalLM,
    config: Config,
    checkpoint: bool,
    to_remove: List[int],
):
    """
    Returns the NHS dataset train/test dataloaders
    """
    # TODO: fix function
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
    dist.barrier()
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

    dist.barrier()

    test_dataset = test_dataset.map(
        lambda example: llama_two_nhs_conversion(example["text"], tokenizer),
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )
    test_dataset = test_dataset.add_column("id", [i for i in range(len(test_dataset))])

    dist.barrier()

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

    dist.barrier()

    if to_remove != []:
        to_remove = set(to_remove)

        train_dataset = train_dataset.filter(
            lambda example: remove_unwanted_rows(example, to_remove),
            batched=True,
            num_proc=8,
            batch_size=5000,
        )

    print(f"Train dataset length {len(train_dataset)} (after packing)")
    print(f"Eval dataset length {len(test_dataset)} (no packing)")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=dc, batch_size=train_bs
    )

    eval_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=dc, batch_size=eval_bs
    )

    return orig_length, train_dataloader, eval_dataloader