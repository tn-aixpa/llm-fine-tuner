import json
import torch
import huggingface_hub
import wandb
import utils
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_dataset
import sys

def train(
    hf_token: str = None,
    model_id: str,
    from_base: int,
    train_data_path: str,
    dev_data_path: str,
    output_dir: str,
    final_dir: str,
    project_name: str,
    run_name: str,
    quantization: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    max_sequence_length: int,
    early_stopping_patience: int,
    learning_rate: float,
    scheduler_type: str,
    train_batch_size: int,
    eval_batch_size: int,
    grad_accum_steps: int,
    num_epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    wandb_key: str = None
) -> None:
    """
    Train the LLM model with the given dataset and configuration.

    Args:
        hf_token (str): Hugging Face API token. Required only for private models; not needed when using public models.
        model_id (str): Model ID
        from_base (int): From Base? (0 or 1)
        train_data_path (str): Training dataset path
        dev_data_path (str): Dev dataset path
        output_dir (str): Output directory for model and checkpoints
        final_dir (str): Output directory for final model
        project_name (str): Project name for logging
        run_name (str): Run name for logging
        quantization (int): Quantization: 0 if none, 4 for 4bit quantization
        lora_rank (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout rate
        max_sequence_length (int): Maximum sequence length
        early_stopping_patience (int): Early stopping patience
        learning_rate (float): Learning rate for training
        scheduler_type (str): Learning rate scheduler type
        train_batch_size (int): Training batch size
        eval_batch_size (int): Evaluation batch size
        grad_accum_steps (int): Gradient accumulation steps
        num_epochs (int): Number of training epochs
        weight_decay (float): Weight decay for optimizer
        warmup_ratio (float): Warmup ratio for learning rate schedule
        logging_steps (int): Number of steps between logging
        eval_steps (int): Number of steps between evaluations
        save_steps (int): Number of steps between model checkpoints
        wandb_key (str): Weights & Biases API key
    """
    # Logging in Hugging Face and WandB
    if hf_token is not None:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            print(f"Error logging into Hugging Face. Check your token.")
            sys.exit(1)
        
    if wandb_key is not None:
        try:
            wandb.login(key=wandb_key)
            wandb.init(project=project_name, name=run_name)
        except Exception as e:
            print(f"Error logging into wanddb. Check your key.{e}")
            sys.exit(1)

    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

    # Loading the model, applying quantization if requested
    if quantization == 4:
        print("4bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("No quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Checking if GPU is available
    print("Model is using:", model.device)
    if not torch.cuda.is_available():
        raise Exception("GPU not available")

    # Apply LoRA adapters
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
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

    # Load datasets
    # with open(train_data_path, "r") as file:
    #     data = json.load(file)
    #     train_dataset = Dataset.from_list(data)
    #     train_dataset = train_dataset.shuffle(seed=42)
        
    # train_dataset = utils.prepare_dataset(data=data, tokenizer=tokenizer, from_base=False if args.from_base==0 else True, guidelines=guidelines if args.guidelines==1 else None, previous_messages=args.previous_messages).shuffle(seed=42)

    # with open(dev_data_path, "r") as file:
    #     data = json.load(file)
    #     dev_dataset = Dataset.from_list(data)
    #     dev_dataset = dev_dataset.shuffle(seed=42)

    # dev_dataset = utils.prepare_dataset(data=data, tokenizer=tokenizer, from_base=False if args.from_base==0 else True, guidelines=guidelines if args.guidelines==1 else None, previous_messages=args.previous_messages).shuffle(seed=42)

    try:
        dataset = load_dataset("LanD-FBK/AIxPA_Dialogue_Dataset", data_files={
        "train": train_data_path,
        "validation": dev_data_path,
        })
        train_dataset = dataset["train"].shuffle(seed=42)
        dev_dataset = dataset["validation"].shuffle(seed=42)
    except Exception as e:
        print(f"Error loading dataset. Check your data paths.")
        sys.exit(1)


    # Setting training arguments
    max_seq_length = max_sequence_length
    early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[early_stopping],
        args=SFTConfig(
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            report_to="wandb",
            learning_rate=learning_rate,
            lr_scheduler_type=scheduler_type,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=num_epochs,
            logging_steps=logging_steps,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            output_dir=output_dir,
            seed=0,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    # Launch training and save best model
    trainer.train()
    model.save_pretrained(final_dir)

# if __name__ == "__main__":
#     # For local testing/debugging
#     train(
#         hf_token=" ",
#         wandb_key=" ",
#         model_id="meta-llama/Llama-3.1-8B-Instruct",
#         from_base=0,
#         train_data_path="data_AmiciFamiglia_only_ground/train.json",
#         dev_data_path="data_AmiciFamiglia_only_ground/validation.json",
#         output_dir="checkpoints/Llama-3.1-8B-Instruct/AmiciFamiglia_only_ground",
#         final_dir="weights/Llama-3.1-8B-Instruct/run_AmiciFamiglia_only_groundfamily_ground",
#         project_name="AmiciFamiglia_only_groundd",
#         run_name="Llama-3.1-8B-Instruct_AmiciFamiglia_only_ground",
#         quantization=4,
#         lora_rank=32,
#         lora_alpha=64,
#         lora_dropout=0,
#         max_sequence_length=2300,
#         early_stopping_patience=10,
#         learning_rate=5e-5,
#         scheduler_type="cosine",
#         train_batch_size=2,
#         eval_batch_size=2,
#         grad_accum_steps=3,
#         num_epochs=5,
#         weight_decay=0.01,
#         warmup_ratio=0.03,
#         logging_steps=20,
#         eval_steps=20,
#         save_steps=20,
#     )
