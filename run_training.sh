#!/bin/bash

model="Llama-3.1-8B"

# Base model or Instruct model
FROM_BASE=0

# Modify based on base model
if [ "$FROM_BASE" -eq 0 ]; then
    model="${model}-Instruct"
fi

# Hugging Face API token
HF_TOKEN=" " # ADD YOUR TOKEN
WANDB_KEY=" " # ADD YOUR TOKEN

# Model and dataset paths
MODEL_ID="meta-llama/${model}"
HF_DATASET_NAME="LanD-FBK/AIxPA_Dialogue_Dataset"
TRAIN_DATA_PATH="data_AmiciFamiglia_only_ground/train.json"
DEV_DATA_PATH="data_AmiciFamiglia_only_ground/validation.json"

# Quantization (0, 4)
QUANTIZATION=4

# LoRA (Low-Rank Adaptation) configuration
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0

# Training parameters
MAX_SEQUENCE_LENGTH=2300
EARLY_STOPPING_PATIENCE=10
LEARNING_RATE=5e-5
SCHEDULER_TYPE="cosine"
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
GRAD_ACCUM_STEPS=3
NUM_EPOCHS=5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.03

# Logging and checkpointing
LOGGING_STEPS=20
EVAL_STEPS=20
SAVE_STEPS=20

# Output and run name
OUTPUT_DIR="checkpoints/${model}/AmiciFamiglia_only_ground"
FINAL_DIR="weights/${model}/run_AmiciFamiglia_only_groundfamily_ground"
PROJECT_NAME="AmiciFamiglia_only_groundd"
RUN_NAME="${model}_AmiciFamiglia_only_ground"

# Modify based on base model
if [ "$FROM_BASE" -eq 1 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}"
    FINAL_DIR="${FINAL_DIR}"
    RUN_NAME="${RUN_NAME}"
    PROJECT_NAME="Precrisis-SFT-From-Base_2"
fi

clear

# Echoing the parameters for visibility
echo "Launching training with the following parameters:"
echo "  Model ID: $MODEL_ID"
echo "  HF Dataset Name: $HF_DATASET_NAME"
echo "  Train Data Path: $TRAIN_DATA_PATH"
echo "  Dev Data Path: $DEV_DATA_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Final Model Directory: $FINAL_DIR"
echo "  Project Name: $PROJECT_NAME"
echo "  Run Name: $RUN_NAME"
echo "  Quantization: $QUANTIZATION"
echo "  LoRA Rank: $LORA_RANK"
echo "  LoRA Alpha: $LORA_ALPHA"
echo "  LoRA Dropout: $LORA_DROPOUT"
echo "  Max Sequence Length: $MAX_SEQUENCE_LENGTH"
echo "  Early Stopping Patience: $EARLY_STOPPING_PATIENCE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Scheduler Type: $SCHEDULER_TYPE"
echo "  Train Batch Size: $TRAIN_BATCH_SIZE"
echo "  Eval Batch Size: $EVAL_BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Warmup Ratio: $WARMUP_RATIO"
echo "  Logging Steps: $LOGGING_STEPS"
echo "  Eval Steps: $EVAL_STEPS"
echo "  Save Steps: $SAVE_STEPS"

# Set PYTHONPATH to include /data (mapped from $PWD)
export PYTHONPATH=/data:$PYTHONPATH

# Execute the train() function directly
python -c "from Llama_sft_training import train; train(\
    hf_token='$HF_TOKEN',\
    wandb_key='$WANDB_KEY',\
    model_id='$MODEL_ID',\
    from_base=$FROM_BASE,\
    hf_dataset_name='$HF_DATASET_NAME',\
    train_data_path='$TRAIN_DATA_PATH',\
    dev_data_path='$DEV_DATA_PATH',\
    output_dir='$OUTPUT_DIR',\
    final_dir='$FINAL_DIR',\
    project_name='$PROJECT_NAME',\
    run_name='$RUN_NAME',\
    quantization=$QUANTIZATION,\
    lora_rank=$LORA_RANK,\
    lora_alpha=$LORA_ALPHA,\
    lora_dropout=$LORA_DROPOUT,\
    max_sequence_length=$MAX_SEQUENCE_LENGTH,\
    early_stopping_patience=$EARLY_STOPPING_PATIENCE,\
    learning_rate=$LEARNING_RATE,\
    scheduler_type='$SCHEDULER_TYPE',\
    train_batch_size=$TRAIN_BATCH_SIZE,\
    eval_batch_size=$EVAL_BATCH_SIZE,\
    grad_accum_steps=$GRAD_ACCUM_STEPS,\
    num_epochs=$NUM_EPOCHS,\
    weight_decay=$WEIGHT_DECAY,\
    warmup_ratio=$WARMUP_RATIO,\
    logging_steps=$LOGGING_STEPS,\
    eval_steps=$EVAL_STEPS,\
    save_steps=$SAVE_STEPS\
)"
