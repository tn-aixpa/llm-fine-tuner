# How to Install Dependencies and Run Training

The training is initiated via the `train()` function in [`Llama_sft_training.py`](./Llama_sft_training.py).


## 🔧 Prerequisites

- Python: Version 3.8 or higher.

- GPU: Required for training.
  
- API Tokens:

    - Hugging Face token with access to Llama-3.1-8B-Instruct.Create one [here](https://huggingface.co/settings/tokens). If you want to use public model, you don't need this parameter.

    - Weights & Biases API key for logging (optional).Create one [here](https://wandb.ai/home). This parameter is optional.


## ⚙️ Setup

**Install Dependencies:**

```python
pip install -r requirements.txt
```


**Configure API Tokens:** Set environment variables or update `run_training.sh`:

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_KEY="your_wandb_key"
```
## 🐳 Docker (Optional) 
For a reproducible GPU-enabled environment:
**1. Build the Docker Image:**

```bash
./build.sh
```


**2. Run the Container:**

```bash
./docker_run.sh
```


**3. Monitor Logs:**

```bash
docker logs aixpa-training-0
```
 > [!NOTE] 
> After running the docker it will show __the logs will not show any update for some minutes__ while it is downloading llama 3.1 model from huggingface
> To check it the model is downloading, it is saved within the docker in `/root/.cache/huggingface` (the script needs to download about 15GB)

## 🚀 Running the Project

Use the `train()` function in `Llama_sft_training.py`:

```python
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
)
```

🧪 Example Usage

If you want to run locally, Update run_training.sh with your paths and parameters:

```bash
HF_TOKEN="your_hf_token"
WANDB_KEY="your_wandb_key"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET_NAME="LanD-FBK/AIxPA_Dialogue_Dataset",
TRAIN_DATA_PATH="l_data/data_AmiciFamiglia_only_ground/train.json"
DEV_DATA_PATH="l_data/data_AmiciFamiglia_only_ground/validation.json"
OUTPUT_DIR="checkpoints/Llama-3.1-8B-Instruct/AmiciFamiglia_only_ground"
FINAL_DIR="weights/Llama-3.1-8B-Instruct/run_AmiciFamiglia_only_ground"
PROJECT_NAME="AmiciFamiglia_only_ground"
RUN_NAME="Llama-3.1-8B-Instruct_AmiciFamiglia_only_ground"
QUANTIZATION=4
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0
MAX_SEQUENCE_LENGTH=2300
```

Launch training:

```bash
./run_training.sh
```

- **Outputs**

    - **Checkpoints:** Saved to `checkpoints/<model>/<run_name>/` every `save_steps`.

    - **Best Model:** Saved to `weights/<model>/<run_name>_best/` based on lowest `eval_loss`.(unless renamed in `run_training.sh`)


## 📝 Additional Notes
- **Quantization:** Set `QUANTIZATION=4` for 4-bit training/inference.

- **LoRA Parameters:** Adjust `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` for fine-tuning efficiency.

- **WandB Logging:** Enabled if `WANDB_KEY` is provided.

- **Early Stopping:** Configurable via `early_stopping_patience`.
