# LLM Fine Tuner

LLM Fine Tuner represents a transformers-based implementation of the fine-tuning algorithm
aiming to produce a LoRA adapter based on the dialog-like supporting dataset.

## 🔧 Prerequisites

- Python: Version 3.8 or higher.

- GPU: Required for training.
  
- API Tokens:

    - Hugging Face token with access to Llama-3.1-8B-Instruct.Create one [here](https://huggingface.co/settings/tokens). If you want to use public model, you don't need this parameter.

    - Weights & Biases API key for logging (optional).Create one [here](https://wandb.ai/home). This parameter is optional.

## Hardware Requirements

The training procedure requires a certain model of space (depends on the base model, dataset, and number of checkpoints) and GPU. The suggested VRAM amount is 80G

## Train Dataset

To perform the training, the dataset should be structured in a specific manner, where the entries represent the dialogs. See [here](./howto/data.md) for the details of data preparation. The dataset should be made available to training through [HuggingFace](https://huggingface.co/).

## Training parameters

The training script accepts the following parameters (all must be provided unless marked as optional):

| Argument                  | Type    | Description                                                                                 |
|---------------------------|---------|---------------------------------------------------------------------------------------------|
| --model_id                | str     | Model ID (e.g., huggingface repo/model name)                                                |
| --from_base               | int     | Whether to start from base model (0 or 1)                                                   |
| --hf_dataset_name         | str     | Name of the dataset on Hugging Face Hub                                                     |
| --train_data_path         | str     | Path to the training dataset                                                                |
| --dev_data_path           | str     | Path to the dev/validation dataset                                                          |
| --output_dir              | str     | Output directory for model and checkpoints                                                  |
| --final_dir               | str     | Output directory for the final model                                                        |
| --project_name            | str     | Project name for logging (e.g., for Weights & Biases)                                       |
| --run_name                | str     | Run name for logging                                                                        |
| --quantization            | int     | Quantization: 0 for none, 4 for 4-bit quantization                                          |
| --lora_rank               | int     | LoRA rank                                                                                   |
| --lora_alpha              | int     | LoRA alpha                                                                                  |
| --lora_dropout            | float   | LoRA dropout rate                                                                           |
| --max_sequence_length     | int     | Maximum sequence length                                                                     |
| --early_stopping_patience | int     | Early stopping patience                                                                     |
| --learning_rate           | float   | Learning rate                                                                               |
| --scheduler_type          | str     | Learning rate scheduler type                                                                |
| --train_batch_size        | int     | Training batch size                                                                         |
| --eval_batch_size         | int     | Evaluation batch size                                                                       |
| --grad_accum_steps        | int     | Gradient accumulation steps                                                                 |
| --num_epochs              | int     | Number of training epochs                                                                   |
| --weight_decay            | float   | Weight decay for optimizer                                                                  |
| --warmup_ratio            | float   | Warmup ratio for learning rate schedule                                                     |
| --logging_steps           | int     | Number of steps between logging                                                             |
| --eval_steps              | int     | Number of steps between evaluations                                                         |
| --save_steps              | int     | Number of steps between model checkpoints                                                   |
| --log_model               | flag    | Whether the model should be logged to the platform                                          |
| --model_name              | str     | Name of the model to be logged if --log_model is specified                                  |
|---------------------------|---------|---------------------------------------------------------------------------------------------|

Example usage:

```bash
python [main.py](../src/main.py) \
  --model_id=<MODEL_ID> \
  --from_base=1 \
  --hf_dataset_name=<DATASET_NAME> \
  --train_data_path=<TRAIN_PATH> \
  --dev_data_path=<DEV_PATH> \
  --output_dir=<OUTPUT_DIR> \
  --final_dir=<FINAL_DIR> \
  --project_name=<PROJECT_NAME> \
  --run_name=<RUN_NAME> \
  --quantization=4 \
  --lora_rank=32 \
  --lora_alpha=64 \
  --lora_dropout=0.05 \
  --max_sequence_length=2048 \
  --early_stopping_patience=10 \
  --learning_rate=5e-5 \
  --scheduler_type=cosine \
  --train_batch_size=2 \
  --eval_batch_size=2 \
  --grad_accum_steps=3 \
  --num_epochs=5 \
  --weight_decay=0.01 \
  --warmup_ratio=0.03 \
  --logging_steps=20 \
  --eval_steps=20 \
  --save_steps=20
  ```
  Replace values in angle brackets with your actual configuration. 

  ## Execution modalities

  The traning may be performed in different modalities:
  
  - as a [docker container](./howto/train.md)
  - as a [python job in the platform](./howto/train_container.md)

- **Outputs**

    - **Checkpoints:** Saved to the specified path (``output_dir``) every `save_steps`.

    - **Best Model:** Saved to the specified path  (``final_dir``) based on lowest `eval_loss`.

## 📝 Additional Notes
- **Quantization:** Set `QUANTIZATION=4` for 4-bit training/inference.

- **LoRA Parameters:** Adjust `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` for fine-tuning efficiency.

- **WandB Logging:** Enabled if `WANDB_KEY` is provided.

- **Early Stopping:** Configurable via `early_stopping_patience`.
