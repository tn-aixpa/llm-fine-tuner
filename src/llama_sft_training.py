import json
import torch
import huggingface_hub
import wandb
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
    model_id: str,
    from_base: int,
    hf_dataset_name: str,
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
    hf_token: str = None,
    wandb_key: str = None
) -> None:
    """
    Train the LLM model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        from_base (int): From Base? (0 or 1)
        hf_dataset_name (str): Name of the dataset on Hugging Face Hub.
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
        hf_token (str): Hugging Face API token. Required only for private models; not needed when using public models.
        wandb_key (str): Weights & Biases API key
    """
    # Logging in Hugging Face and WandB
    if hf_token is not None:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            raise RuntimeError("Error logging into Hugging Face. Check your token.")
        
    if wandb_key is not None:
        try:
            wandb.login(key=wandb_key)
            wandb.init(project=project_name, name=run_name)
        except Exception as e:
            print("WandB not configured")
        
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

    try:
        dataset = load_dataset(hf_dataset_name, data_files={
        "train": train_data_path,
        "validation": dev_data_path,
        })
        train_dataset = dataset["train"].shuffle(seed=42)
        dev_dataset = dataset["validation"].shuffle(seed=42)
    except Exception as e:
        raise RuntimeError("Error loading dataset. Check your data paths.")
        

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
            report_to="wandb" if wandb_key is not None else None,
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

def train_and_log_model(
    project,
    target_model_name: str,
    model_id: str,
    hf_dataset_name: str,
    train_data_path: str,
    dev_data_path: str,
    from_base: int = 0,
    quantization: int = 4,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0,
    max_sequence_length: int = 2300,
    early_stopping_patience: int = 10,
    learning_rate: float = 5e-5,
    scheduler_type: str = "cosine",
    train_batch_size: int = 2,
    eval_batch_size: int = 2,
    grad_accum_steps: int = 3,
    num_epochs: int = 5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    logging_steps: int = 20,
    eval_steps: int = 20,
    save_steps: int = 20,
    hf_token: str = None,
    wandb_key: str = None,
    wandb_project: str = None,
    wandb_run: str = None
    ):
    """
    Train the LLM model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        from_base (int): From Base? (0 or 1)
        hf_dataset_name (str): Name of the dataset on Hugging Face Hub.
        train_data_path (str): Training dataset path
        dev_data_path (str): Dev dataset path
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
        hf_token (str): Hugging Face API token. Required only for private models; not needed when using public models.
        wandb_key (str): Weights & Biases API key (optional)
    """

    output_dir = '/local_data/checkpoints/ground'
    final_dir = 'local_data/weights/ground'    

    hf_token = None
    wandb_key = None
    try:    
        hf_token = project.get_secret("HF_TOKEN").read_secret_value()
    except Exception:
        pass

    try:    
        wandb_key = project.get_secret("WANDB_KEY").read_secret_value()
    except Exception:
        pass
    
    train(
        model_id, 
        from_base, 
        hf_dataset_name,
        train_data_path,
        dev_data_path,
        output_dir,
        final_dir,
        wandb_project,
        wandb_run,
        quantization,
        lora_rank,
        lora_alpha,
        lora_dropout,
        max_sequence_length,
        early_stopping_patience,
        learning_rate,
        scheduler_type,
        train_batch_size,
        eval_batch_size,
        grad_accum_steps,
        num_epochs,
        weight_decay,
        warmup_ratio,
        logging_steps,
        eval_steps,
        save_steps,
        hf_token=hf_token,
        wandb_key=wandb_key
    )

    model_params = {
        "quantization": quantization,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_sequence_length": max_sequence_length,
        "early_stopping_patience": early_stopping_patience,
        "learning_rate": learning_rate,
        "scheduler_type": scheduler_type,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps
        
    }
    
    project.log_model(
        name=target_model_name,
        kind="huggingface",
        base_model=model_id,
        parameters=model_params,
        source=final_dir +"/",
    )      
