Guide on how to lunch codes

# Usage

`pip install -f requirements.txt`

To add all dependencies needed in these files

then create a file named 'config.yaml'

# config.yaml

The `config.yaml` contains the following:

```
hf_token: "HUGGINGFACE_TOKEN"
wandb_token: "WANDB_TOKEN"
```

Replace `HUGGINGFACE_TOKEN` with your Meta Llama 3.1B Instruct's token claimed [in the following HuggingFace page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
Replace `WANDB_TOKEN` with your token for WanDB. This allows to see the progress of the finetuning


# data_cleaner.py

This file is used to create the training data from the starting dialogue JSON files and the matching documents.

To launch the script write in console:

```python3 llm_data_preparation.py```


# Llama_sft_training.py and sft_training.sh

With these, it is possible to start the finetuning of the Llama model.

all the models hyperparameters can be edited within `sft_training.sh`.

To launch the finetuning, write in console:

```
chmod -x sft_training.sh
./sft_training.sh
```

# interactive_model_interrogation.ipynb

This notebook allows to test and use interactively the finetuned model.

To launch, ensure that you have Jupyter notebook installed. If you do not have it, write

```pip install notebook```

You can then launch the following command

```jupyter notebook```

This will open a page in your browser. Navigate to the location where `interactive_model_interrogation.ipynb` is stored and open it. From there, execute each cell and follow any written instruction.