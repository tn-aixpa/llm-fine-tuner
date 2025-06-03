import json
import torch
import huggingface_hub
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import os
import sys

def calculate_token_stats(
    dataset_path: str,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    hf_token: str = None
) -> tuple[int, float]:
    """
    Calculate total and average number of tokens in the dataset.

    Args:
        dataset_path (str): Path to the JSON dataset file
        model_id (str): Model ID for the tokenizer (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
        hf_token (str, optional): Hugging Face API token for authentication

    Returns:
        tuple[int, float]: (total_tokens, average_tokens)
    """
    # Login to Hugging Face if token is provided
    if hf_token:
        try:
            huggingface_hub.login(hf_token)
        except Exception as e:
            print(f"Error logging in to Hugging Face. Check your token.")
            sys.exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path, "r") as file:
        data = json.load(file)
        dataset = Dataset.from_list(data)

    # Calculate tokens
    tokens_count = 0
    lengths = []
    for instance in tqdm(dataset, desc="Processing dataset"):
        input_id = tokenizer(instance["text"], return_tensors="pt", truncation=True)
        token_len = len(input_id["input_ids"][0])
        tokens_count += token_len
        lengths.append(token_len)

    average_tokens = tokens_count / len(dataset) if len(dataset) > 0 else 0

    # Print results (for debugging/logging)
    print(f"Dataset: {dataset_path}")
    print(f"Sample text: {dataset[6]['text']}")
    print(f"Token lengths: {lengths}")
    print(f"Total number of tokens: {tokens_count}")
    print(f"Average number of tokens: {average_tokens}")

    return tokens_count, average_tokens

#     # For local testing/debugging
# if __name__ == "__main__":
#     dataset_path = "data/train_data.json"
#     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#     hf_token = "" #Insert your huggingface token
#     total, avg = calculate_token_stats(dataset_path, model_id, hf_token)
