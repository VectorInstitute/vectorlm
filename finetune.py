import time
import argparse
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
import bitsandbytes as bnb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from datasets import load_dataset

from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="HuggingFace path or model path")
    parser.add_argument("--full_finetune", default=False, type=bool, help="Finetune all parameters instead of LoRA")
    parser.add_argument("--bits", default=8, type=int, help="bitwidth for weight quantization")
    parser.add_argument("--bf16", default=True, type=bool, help="perform computation in bf16")
    parser.add_argument("--fp16", default=False, type=bool, help="perform computation in fp16")
    parser.add_argument("--quant_type", default="nf4", type=str, help="fp4 or nf4 weight quantization type")
    parser.add_argument("--double_quant", default=True, type=bool, help="quantize the quantization parameters")
    parser.add_argument("--gradient_checkpointing", default=False, type=bool, help="enable grad checkpoint to reduce memory")
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


def get_accelerate_model(args):
    # setup model
    compute_dtype = (torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.fp32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map="auto",
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=compute_dtype

    # LoRA prep
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        # TODO add lora weight checkpoint load & save https://github.com/artidoro/qlora/blob/main/qlora.py
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
            
    print_trainable_parameters(args, model)
    # model = torch.compile(model) # doesnt work
    # TODO add flash attention
    return model

if __name__ == "__main__":
    args = parse_args()
    model = get_accelerate_model(args)
    # BATCH_SIZE = 16
    # SEQ_LEN = 128
    # batch = {
    #     "input_ids": torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)),
    #     "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN),
    #     "labels": torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)),
    # }
    
    # start = time.time()
    # for i in tqdm(range(100)):
        # with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        #     out = model(**batch) # TODO add in trainer
        #     loss = out.loss
        #     loss.backward()

    # TODO llama-2 must be trained in bfloat16 https://huggingface.co/docs/transformers/model_doc/llama2
    # TODO try trainer
    # 1. use the downloaded models
    # 2. try to pass that through trainer
    
    dataset = load_dataset("imdb", split="train")
    # # https://huggingface.co/docs/trl/sft_trainer
    # # TODO add custom training args
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b") # https://github.com/huggingface/transformers/issues/22762
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir="delete_this",
        per_device_train_batch_size=2, # num_gpus x per_device_batch_size
    )
    trainer = SFTTrainer(
        model, # can pass in peft_config, but how do we allow compilation? might need to write our own trainer
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=128,
    )

    # # TODO make DDP trainer, FSDP trainer
    with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        trainer.train() 