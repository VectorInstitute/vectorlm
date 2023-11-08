import os
import sys
import argparse
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import bitsandbytes as bnb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from peft.tuners.lora import LoraLayer
from datasets import load_dataset
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("--model_name_or_path", type=str, help="HuggingFace path or model path")
    parser.add_argument("--full_finetune", default=False, type=bool, help="Finetune all parameters instead of LoRA")
    parser.add_argument("--max_seq_length", default=1024, type=int, help="Maximum sequence length of text")
    parser.add_argument("--batch_size", default=128, type=int, help="Total batch size desired.")
    parser.add_argument("--batch_size_per_device", default=16, type=int, help="The maximum batch size per gpu.")
    parser.add_argument("--epochs", default=1.0, type=float, help="Number of training epochs.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Peak learning rate")
    parser.add_argument("--warmup_ratio", default=0.10, type=float, help="Ratio of steps for linear warmup.")
    parser.add_argument("--scheduler", default="cosine", type=str, help="Learning rate scheduler.")

    # saving/logging parameters
    parser.add_argument("--output_dir", type=str, help="Output dir for logs and checkpoints")
    parser.add_argument("--save_steps", type=str, default=10, help="Frequency to save model checkpoint")
    parser.add_argument("--logging_steps", type=int, default=1, help="Frequency of logging")
    parser.add_argument("--report_to", type=str, default="wandb", help="Cloud logging service")

    # lora/performance optimization parameters
    parser.add_argument("--bits", default=4, type=int, help="bitwidth for weight quantization")
    parser.add_argument("--bf16", default=True, type=bool, help="perform computation in bf16")
    parser.add_argument("--fp16", default=False, type=bool, help="perform computation in fp16")
    parser.add_argument("--quant_type", default="nf4", type=str, help="fp4 or nf4 weight quantization type")
    parser.add_argument("--double_quant", default=True, type=bool, help="quantize the quantization parameters")
    parser.add_argument("--gradient_checkpointing", default=True, type=bool, help="enable rematerialization to reduce memory")
    parser.add_argument("--use_flash_attention_2", default=True, type=bool, help="enable flash attention 2 in transformers >=4.34")
    parser.add_argument("--lora_r", default=64, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha")
    args = parser.parse_args()
    return args


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param:.2f}%"
    )


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(args, checkpoint_dir):
    compute_dtype = (torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.fp32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map={'': int(os.environ["LOCAL_RANK"])}, # disable pipeline parallelism, init model on each device separately
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=args.bits == 16,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
    )
    # disable horizontal model parallelism for DP training
    setattr(model, 'model_parallel', False)
    setattr(model, 'is_parallelizable', False)
    model.config.torch_dtype=compute_dtype

    # LoRA prep
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if checkpoint_dir is not None:
            if int(os.environ["LOCAL_RANK"]) == 0:
                print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            modules = find_all_linear_names(args, model)
            lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=modules,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, lora_config)

        # force some activations in higher bitwidth for stability
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print_trainable_parameters(args, model)
    return model


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if int(os.environ["LOCAL_RANK"]) == 0:
            print('Saving PEFT checkpoint...')
            if state.best_model_checkpoint is not None:
                checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
            else:
                checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = os.path.join(checkpoint_dir, f'checkpoint-{max_step}')
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training


if __name__ == "__main__":
    args = parse_args()
    # calculate batch size
    assert args.batch_size % args.batch_size_per_device == 0
    num_devices = int(os.environ["WORLD_SIZE"])
    gradient_accumulation_steps = args.batch_size // (args.batch_size_per_device * num_devices)
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')
        sys.exit(0)
    model = get_accelerate_model(args, checkpoint_dir) # QLoRA model
    os.environ["WANDB_PROJECT"] = "<my-amazing-project>" # name your W&B project 
    os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints
    # tokenizer = get_tokenizer(args)

    # finetuning: /checkpoint/opt_test/original/clinical_llm/datasets/
    # https://huggingface.co/datasets/ywchoi/mtb_final_ordered
    # https://huggingface.co/datasets/ywchoi/mdpi
    # instruction tuning: 
    # Eval: use callback: https://github.com/artidoro/qlora/blob/main/qlora.py#L747 <- example

    #train_dataset, eval_dataset = get_dataset()
    train_dataset = load_dataset("imdb", split="train") # dummy dataset
    eval_dataset = load_dataset("imdb", split="test") 

    # we have incorrect tokenizer class for llama checkpoints, load the correct one to suppress errors
    # https://github.com/huggingface/transformers/issues/22762
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b") 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        tf32=True,
        bf16=args.bf16,
        fp16=args.fp16,
        bf16_full_eval=args.bf16,
        fp16_full_eval=args.fp16,
        lr_scheduler_type=args.scheduler,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        resume_from_checkpoint=(checkpoint_dir is not None),
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    trainer.train(resume_from_checkpoint=(checkpoint_dir is not None))

    # Notes
    # 1. LLaMA/LLaMA-2 must be trained in bfloat16: https://huggingface.co/docs/transformers/model_doc/llama2
    # 2. 8 bit adam can be used to save 7-14% x num_parameters of GPU memory: https://huggingface.co/docs/transformers/perf_train_gpu_one  
    # 3. Different types of trainer: https://huggingface.co/docs/trl/sft_trainer

    # Throughput
    # 7B
    # 4 A100 x 32 x 1024: 10k tokens/s (13s/iteration)

    # 30B
    # 4 A100 x 16 x 1024: 2600 tokens/s (25s/iteration)
    

    