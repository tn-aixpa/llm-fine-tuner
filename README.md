# LLM Finetuner

A lightweight framework for fine-tuning [`Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), designed for efficient and reproducible training.

AIxPA

- ``kind``: 
- ``ai``: NLP
- ``domain``:  PA

 The LLM Finetuner supports:

- **Efficient Fine-Tuning:** Utilizes **LoRA (Low-Rank Adaptation)** for parameter-efficient training.
- **4-Bit Quantization:** Reduces memory footprint for training and inference.
- **JSON Dataset Support:** Processes datasets in JSON format with Llama-3.1 chat template.
- **Docker Support:** Optional GPU-enabled Docker container for reproducible environments.
- **WandB Integration:** Tracks training metrics with Weights & Biases.


## Usage

Tool usage documentation [here](./docs/usage.md).

## How To
- [Data Overview, Dataset Preparation and Token Count Analysis.](./docs/howto/data.md)
- [Train LLM Adapter with the platform](./docs/howto/train.md)
- [Train with Docker container](./docs/howto/train_container.md)


## License

[Apache License 2.0](./LICENSE)
