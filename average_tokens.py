import json
import torch
import huggingface_hub
from transformers import (
    AutoTokenizer
)
from datasets import Dataset
from tqdm import tqdm
import huggingface_hub
import os

def main():
    huggingface_hub.login("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")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

    with open("data/train_data.json", "r") as file:
        data = json.load(file)
        train_dataset = Dataset.from_list(data)

    print(train_dataset[6]["text"])

    tokens_count = 0
    lengths = [] 

    for instance in tqdm(train_dataset):
        input_id = tokenizer(instance["text"], return_tensors="pt", truncation=True)
        tokens_count += len(input_id["input_ids"][0])
        lengths.append(len(input_id["input_ids"][0]))

    print(lengths)

    print("Total number of tokens", tokens_count)

    print("Average numeber of tokens", tokens_count/len(train_dataset))


if __name__ == "__main__":
    main()
