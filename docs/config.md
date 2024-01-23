# Configuration
`config.yaml` as it stands is a sample of what gets used when you run `examples/llama_example.py`. This is what has been used in the making of this framework and our research projects otherwise. Please find the configurable options below.

* `model`: The directory containing the model and tokenizer. It assumes that the model and tokenizer files are available in this directory. On the Vector cluster, you can find public models stored under `/model-weights`.
* `enable_wandb_logging`: Whether you would like to use w&b for logging.

## Weights & Biases Configuration

The key-value pairs stored under `wandb_config` are directly passed into the [`wandb.init`](https://docs.wandb.ai/ref/python/init) method during the execution of the script. These are spell-sensitive. The whole config file is also logged as part of the run.

## Training Parameters

* `output_dir`: The directory which stores checkpointed states and the final consolidated model.
* `max_seq_len`: The maximum sequence length being used during training. This should be less than or equal to the model's maximum possible sequence length.
* `epochs`: The number of epochs to train over.
* `seed`: The seed number. All devices will be set to this seed before anything.

### Sharding Strategy

* `sharding_strategy`: The FSDP sharding strategy being used. This should be from one of the options available in [PyTorch's FSDP module](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy). For the information below, `X = {model parameters, gradients, optimizer states, activations}`.

    * If you're training on a single node and `X` can fit on a single GPU, then use `NO_SHARD`. This is essentially just Distributed Data Parallel and you will just be parallelizing data batches across GPUs.
    * If you're training on a single node and `X` cannot fit on a single GPU, we recommend using `FULL_SHARD`. This is regular FSDP and `X` (except for activations) is evenly sharded across GPUs. Activations are local to each GPU and are unsharded in FSDP. Check the pseudocode [here](https://engineering.fb.com/2021/07/15/open-source/fsdp/) for more information.
    * If you're training on multiple nodes, we recommend using `HYBRID_SHARD`. This is essentially `FULL_SHARD` within a node, and replication across nodes (DDP). The benefit of this is that the expensive all-gathers and reduce-scatters from FSDP happen over much faster connectivity within nodes (intranode), and the communication between nodes (internode) is limited to all-reduces for gradients.

### Memory & Compute

* `use_mp`: Whether to use mixed precision. This is done using bf16.
* `use_activation_checkpointing`: Whether to use activation checkpointing. This greatly reduces memory footprint as only a few intermediate activations as saved during the forward pass, and are then recomputed for the backward pass on the fly. However, the tradeoff between compute vs. memory usually makes this worth it.
* `use_flash_attention`: Whether to use Flash Attention. If it is supported for your model in HuggingFace, you can enable this option.

### Gradient

* `max_grad_norm`: The maximum gradient norm used for gradient clipping.
* `gradient_accumulation_steps`: The number of  training steps that we accumulate gradients for in order to achieve a certain global batch size. `global_batch_size = n_gpus * micro_batch_size * gradient_accumulation_steps`. Note that micro-batch size is synonymous with the batch size used per GPU.

### Optimizer

Similar to the wandb config above, these keyword parameters are fed directly into a user-defined optimizer. As such, they are spell-sensitive. See `examples/llama_example.py` for an example.

### Scheduler

* `lr_scheduler_type`: This can either be our custom defined `plataeu-with-warmup` or an option from what HuggingFace offers as seen [here](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType) (check the enumerations of `transformers.SchedulerType` for what can be passed into here). Note that `plataeu-with-warmup` is the normal reduce-on-plataeu scheduler, but it starts off with a linear warmup for a given number of training steps.
* `warmup_ratio`: The ratio of the total number of training steps that are spent for a linear warmup. This should be between 0 and 1.

### Checkpointing & Logging

* `checkpointing_enabled`: Whether to enable state checkpointing during training.
* `logging_steps`: How often evaluation is run using the evaluation dataset.
* `save_frequency`: The frequency at which checkpointing occurs. This must be between 0 and 1.

## Dataset

* `ignore_index`: The integer index used to ignore a given token in the loss calculation. Cross-entropy loss by default uses `-100`.
* `eval_bs`: Per GPU evaluation batch size.
* `train_bs`: Per GPU training batch size.
* `train_ds`: Path to the preprocessed training dataset.
* `eval_ds`: Path to the preprocessed evaluation dataset.

## Dataset Preprocessing
* `ignore_index`: The integer index used to ignore a given token in the loss calculation. Cross-entropy loss by default uses `-100`.
* `dataset_format`: Here for forward-compatibility.
* `data_field`: The data field that in the dataset that will be used for training.
* `packing_type`: Either `full` or `partial`. `full` packing concatenates the whole dataset and then chunks it. `partial` packing chunks on each individual datapoint (which can span multiple context lengths). **Note:** while packing, sometimes there are multiple tokens that should not be broken up. If they are, then the decoded format ends up being prepended with `##`. There isn't a fix for it yet, but it is something to keep in mind.
* `overlap`: When we chunk a data point during packing, we can choose to have some overlap between the current chunk and the next chunk. This might help the model understand surrounding context during training (although this isn't something we have empirically investigated, we keep this option available to users).
* `add_bos_eos_tokens`: Whether to add `BOS` and `EOS` tokens as defined by the respective HuggingFace tokenizer. If using packing, these will be added after packing is done, so that each chunk of size `max_seq_len` has these tokens.
* `from_disk`: Whether we are going to be loading the dataset to preprocess from disk (the other option is to download straight from HuggingFace).
* `seperator`: If using conditional finetuning (i.e. in a given data point, everything before `separator` will not be used for calculating the loss and its labels will be `ignore_index`). **Note:** if `separator` is not found in a given sequence, the default behavior is that datapoint will be skipped and not be a part of the final set.
* `load_path`: The directory containing the HuggingFace dataset we are loading to preprocess.
* `split`: If `load_path` is a dataset dictionary, `split` specifies which key in this dictionary contains the dataset we are preprocessing.
* `save_path`: The directory we will be saving the processed dataset to.
* `truncate`: Whether or not to truncate (instead of pack) data points that exceed `max_seq_len`.
* `pre_pend`: A possible string we can prepend to our data points before tokenizing.