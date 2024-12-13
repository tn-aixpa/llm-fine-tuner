#!/bin/bash

# Meta-Llama-3.1-8B-Instruct
# Llama-3.2-3B-Instruct
# Llama-3.2-1B-Instruct
model="Llama-3.1-8B"

# Base model or Instruct model
FROM_BASE=0

# Modify based on base model
if [ "$FROM_BASE" -eq 0 ]; then
    model="${model}-Instruct"
fi

# Hugging Face API token
HF_TOKEN=$(grep 'hf_token:' config.yaml | cut -d ' ' -f 2-)
# WanDB API token
WANDB_KEY=$(grep 'wandb_token:' config.yaml | cut -d ' ' -f 2-)

# Model and dataset paths
MODEL_ID="meta-llama/${model}"
TRAIN_DATA_PATH="data/train_data.json"
DEV_DATA_PATH="data/validation_data.json"

# Quantization (0, 4)
QUANTIZATION=4

# LoRA (Low-Rank Adaptation) configuration
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0

# Training parameters
MAX_SEQUENCE_LENGTH=2048
EARLY_STOPPING_PATIENCE=5
LEARNING_RATE=3e-4
SCHEDULER_TYPE="cosine"
TRAIN_BATCH_SIZE=6
EVAL_BATCH_SIZE=6
GRAD_ACCUM_STEPS=1
NUM_EPOCHS=5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.03

# Guidelines and exclude
PREVIOUS_MESSAGES=0
GUIDELINES=0

# Logging and checkpointing
LOGGING_STEPS=1
EVAL_STEPS=50
SAVE_STEPS=50

# Output and run name
OUTPUT_DIR="checkpoints/${model}/run1"
FINAL_DIR="weights/${model}/run1"
PROJECT_NAME="aixpa"
RUN_NAME="${model}_aixpa_run1"

# Modify based on base model
if [ "$FROM_BASE" -eq 1 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}"
    FINAL_DIR="${FINAL_DIR}"
    RUN_NAME="${RUN_NAME}"
    PROJECT_NAME="AIxPA-From-Base"
fi

# Modify based on previous messages
if [ "$PREVIOUS_MESSAGES" -eq 1 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_with_previous_messages"
    FINAL_DIR="${FINAL_DIR}_with_previous_messages"
    RUN_NAME="${RUN_NAME}_with_previous_messages"
fi

# Modify based on guidelines
if [ "$GUIDELINES" -eq 1 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_with_guidelines"
    FINAL_DIR="${FINAL_DIR}_with_guidelines"
    RUN_NAME="${RUN_NAME}_with_guidelines"
fi

clear

# Echoing the parameters for visibility
echo "Launching training with the following parameters:"
echo "  Model ID: $MODEL_ID"
echo "  From base: $FROM_BASE"
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
echo "  Guidelines: $GUIDELINES"
echo "  Previous Messages: $PREVIOUS_MESSAGES"
echo "  Warmup Ratio: $WARMUP_RATIO"
echo "  Logging Steps: $LOGGING_STEPS"
echo "  Eval Steps: $EVAL_STEPS"
echo "  Save Steps: $SAVE_STEPS"

# Execute the Python script with the parameters
python Llama_sft_training.py \
    --hf_token $HF_TOKEN \
    --wandb_key $WANDB_KEY \
    --model_id $MODEL_ID \
    --from_base $FROM_BASE \
    --train_data_path $TRAIN_DATA_PATH \
    --dev_data_path $DEV_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --final_dir $FINAL_DIR \
    --project_name $PROJECT_NAME \
    --run_name $RUN_NAME \
    --quantization $QUANTIZATION \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --max_sequence_length $MAX_SEQUENCE_LENGTH \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --learning_rate $LEARNING_RATE \
    --scheduler_type $SCHEDULER_TYPE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --num_epochs $NUM_EPOCHS \
    --logging_steps $LOGGING_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --guidelines $GUIDELINES \
    --previous_messages $PREVIOUS_MESSAGES \
    --warmup_ratio $WARMUP_RATIO \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS
