import argparse
from llama_sft_training import train
import digitalhub as dh
import os

def upload_model(model_name, project_name, model_id, model_params, src_path): 
    
    print(f"Loading model: {model_name}")
    
    # Crea progetto, togliere local quando useremo backend
    project = dh.get_or_create_project(project_name) # , local=True for testing in local

    project.log_model(
        name=model_name,
        kind="huggingface",
        base_model=model_id,
        parameters=model_params,
        source=src_path + ("/" if src_path[-1] != "/" else "")
    )
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM with SFT and LoRA")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the target model")
    parser.add_argument("--from_base", type=int, required=False, default=0, help="From Base? (0 or 1)")
    parser.add_argument("--hf_dataset_name", type=str, required=True, help="Hugging Face dataset name")
    parser.add_argument("--train_data_path", type=str, required=True, help="Training dataset path")
    parser.add_argument("--dev_data_path", type=str, required=True, help="Dev dataset path")
    parser.add_argument("--output_dir", type=str, required=False, default="/app/local_data/checkpoints/ground", help="Output directory for model and checkpoints")
    parser.add_argument("--final_dir", type=str, required=False, default="/app/local_data/weights/ground", help="Output directory for final model")
    parser.add_argument("--wandb_project_name", type=str, default=None, help="Project name for logging")
    parser.add_argument("--wandb_run_name", type=str,default=None,  help="Run name for logging")
    parser.add_argument("--quantization", type=int, required=False, default=4, help="Quantization: 0 if none, 4 for 4bit quantization")
    parser.add_argument("--lora_rank", type=int, required=False, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, required=False, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, required=False, default=0, help="LoRA dropout rate")
    parser.add_argument("--max_sequence_length", type=int, required=False, default=6500, help="Maximum sequence length")
    parser.add_argument("--early_stopping_patience", type=int, required=False, default=5, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, required=False, default=5e-5, help="Learning rate")
    parser.add_argument("--scheduler_type", type=str, required=False, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--train_batch_size", type=int, required=False, default=3, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, required=False, default=2, help="Evaluation batch size")
    parser.add_argument("--grad_accum_steps", type=int, required=False, default=3, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, required=False, default=10, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, required=False, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, required=False, default=0.03, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, required=False, default=20, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, required=False, default=20, help="Eval steps")
    parser.add_argument("--save_steps", type=int, required=False, default=20, help="Save steps")

    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_KEY")
    project_name = os.environ.get("PROJECT_NAME")

    train(
        model_id=args.model_id,
        from_base=args.from_base,
        hf_dataset_name=args.hf_dataset_name,
        train_data_path=args.train_data_path,
        dev_data_path=args.dev_data_path,
        output_dir=args.output_dir,
        final_dir=args.final_dir,
        project_name=args.wandb_project_name,
        run_name=args.wandb_run_name,
        quantization=args.quantization,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_sequence_length=args.max_sequence_length,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler_type,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        hf_token=hf_token,
        wandb_key=wandb_key
    )

    model_params = {
        "quantization": args.quantization,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_sequence_length": args.max_sequence_length,
        "early_stopping_patience": args.early_stopping_patience,
        "learning_rate": args.learning_rate,
        "scheduler_type": args.scheduler_type,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps
    }

    upload_model(
        model_name=args.model_name,
        project_name=project_name,
        model_id=args.model_id,
        model_params=args.model_params,
        src_path=args.src_path
    )
