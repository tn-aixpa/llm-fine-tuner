import json
import torch
import huggingface_hub
import wandb
import argparse
import utils
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(description="Finetuning parameters")

    # Keys and model id
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face API token')
    parser.add_argument('--wandb_key', type=str, required=True, help='Weights & Biases API key')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID')
    parser.add_argument('--from_base', type=int, default=0, help='From Base?')

    # Dataset paths
    parser.add_argument('--train_data_path', type=str, required=True, help='Training dataset path')
    parser.add_argument('--dev_data_path', type=str, required=True, help='Dev dataset path')

    # Output and run configuration
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for model and checkpoints')
    parser.add_argument('--final_dir', type=str, required=True, help='Output directory for final model')
    parser.add_argument('--project_name', type=str, required=True, help='Project name for logging')
    parser.add_argument('--run_name', type=str, required=True, help='Run name for logging')

    # Quantization
    parser.add_argument('--quantization', type=int, required=True, help='Quantization: 0 if none, 4 for 4bit quantization')

    # LoRA (Low-Rank Adaptation) configuration
    parser.add_argument('--lora_rank', type=int, required=True, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, required=True, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, required=True, help='LoRA dropout rate')

    # Training configuration
    parser.add_argument('--max_sequence_length', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--early_stopping_patience', type=int, required=True, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
    parser.add_argument('--scheduler_type', type=str, required=True, help='Learning rate scheduler type')
    parser.add_argument('--train_batch_size', type=int, required=True, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, required=True, help='Evaluation batch size')
    parser.add_argument('--grad_accum_steps', type=int, required=True, help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay for optimizer')
    parser.add_argument('--warmup_ratio', type=float, required=True, help='Warmup ratio for learning rate schedule')

    # Guidelines and exlude
    parser.add_argument('--guidelines', type=int, default=0, help='If the prompt has guidelines in it')
    parser.add_argument('--previous_messages', type=int, default=0, help='If the prompt has previous messages in it')

    # Logging and checkpointing
    parser.add_argument('--logging_steps', type=int, required=True, help='Number of steps between logging')
    parser.add_argument('--eval_steps', type=int, required=True, help='Number of steps between evaluations')
    parser.add_argument('--save_steps', type=int, required=True, help='Number of steps between model checkpoints')

    args = parser.parse_args()

    # Logging in Hugging Face and WandB
    huggingface_hub.login(token=args.hf_token)
    wandb.login(key=args.wandb_key)

    wandb.init(
        project=args.project_name,
        name=args.run_name
    )

    # Loading the tokenizer for Fine Tuning
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

    # Loading the model, appling quantization if requested
    if args.quantization == 4:
        print("4bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
            )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("No quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    # Checking if GPU is available
    print("Model is using:", model.device)

    if not torch.cuda.is_available():
        raise Exception("GPU not available")
    
    # Apply LoRA adapters
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)

    with open(args.train_data_path, "r") as file:
        data = json.load(file)
        train_dataset = Dataset.from_list(data)

    # train_dataset = utils.prepare_dataset(data=data, tokenizer=tokenizer, from_base=False if args.from_base==0 else True, guidelines=guidelines if args.guidelines==1 else None, previous_messages=args.previous_messages).shuffle(seed=42)

    with open(args.dev_data_path, "r") as file:
        data = json.load(file)
        dev_dataset = Dataset.from_list(data)

    # dev_dataset = utils.prepare_dataset(data=data, tokenizer=tokenizer, from_base=False if args.from_base==0 else True, guidelines=guidelines if args.guidelines==1 else None, previous_messages=args.previous_messages).shuffle(seed=42)

    print(train_dataset[0]["text"])
    
    # Setting training arguments
    max_seq_length = args.max_sequence_length

    # early_stopping = EarlyStoppingCallback(
    #     early_stopping_patience=args.early_stopping_patience
    # )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset    =dev_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        #callbacks=[early_stopping],
        args=TrainingArguments(
            report_to="wandb",
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.scheduler_type,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            num_train_epochs=args.num_epochs,
            logging_steps=args.logging_steps,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            output_dir=args.output_dir,
            seed=0,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        ),
    )

    # Launch training and save best model
    trainer.train()

    model.save_pretrained(args.final_dir)


if __name__ == "__main__":
    main()
