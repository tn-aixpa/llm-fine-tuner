
## Data Overview and Token Count Analysis

The datasets used to train the chatbot are available in different configuration.

AmiciFamiglia and Comuni contain respectively dialogues on Piani Famiglia and generic administrative documents. Folders with all contain both.

All the 3 instances are provided in 2 versions. The first one contains the entire documents on which the dialogues are based on , while the only ground version is smaller and contains only the portions of documents relevant for the dialogue.


## 📝 Dataset Format

Datasets must be JSON files (e.g., `train.json`, `validation.json`) with entries following the Llama-3.1 chat template:

```json
[
  {
    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id>\nYou are a helpful assistant.<|eot_id><|start_header_id|>user<|end_header_id>\nHow do I apply for a birth certificate?<|eot_id><|start_header_id|>assistant<|end_header_id>\nYou can apply at your local registry office...<|eot_id>"
  }
]
```

- **Roles:** `system`, `user`, `assistant`.

- **Tokens:** Must include `<|begin_of_text|>`, `<|eot_id|>`, `<|start_header_id|>`.


🔗 See [Meta-LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) for formatting guidelines.

## 🧹 Dataset Preparation

Use `utils.py` for dataset formatting:

- `prepare_dataset():` Applies Llama-3.1 chat template formatting.

- `format_user_string():` Integrates system instructions, user queries, and assistant responses.

## 📊 Token Analysis

To compute average and total token counts:

```python
python average_tokens.py
```

Or use programmatically:

```python
from average_tokens import calculate_token_stats

total, avg = calculate_token_stats(
    dataset_path="l_data/data_Comuni/train.json",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    hf_token="your_hf_token"
)
```
