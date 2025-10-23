# Train LLM Adapter

It is possible to run the training as a python job on the platform. The following steps may be performed directly from the
jupyter notebook workspace on the platform. Clone the repo within the workspace and perform the following steps.

1. Define the training function

```python

import  digitalhub as dh

project = dh.get_or_create_project("llmpa")

func = project.new_function(
    name="train-llm", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="src/llama_sft_training.py",  
    handler="train_and_log_model",
    requirements=["pandas==2.2.1", "tqdm==4.66.5", "openai==1.8.0", "spacy==3.7.5", "torch==2.5.1", "llama-index==0.9.33", "huggingface-hub>=0.34.0", "rank-bm25==0.2.2", "sentence-transformers==4.1.0", "ranx==0.3.20", "transformers==4.56.1", "openpyxl==3.1.5", "rouge-score==0.1.2", "FlagEmbedding==1.3.4", "wandb==0.19.11", "nltk==3.8.1", "trl==0.20", "bitsandbytes==0.45.5", "datasets==3.6.0", "mpi4py==4.1.0"]

)
```

2. Run the function as job.

It is important to note that for the execution a volume should be created as the space requirements exceed the default available space.

The huggingface token and wandb key should be passed as project secrets or as env variables (not recommended). Create the corresponding secrets (``HF_TOKEN`` and ``WANDB_KEY``) in the project configuration.

```python

train_run = func.run(action="job",
                     profile="1xa100",
                     parameters={
                         "model_id": "meta-llama/Llama-3.1-8B",
                         "model_name": "llmpa",
                         "hf_dataset_name": "LanD-FBK/AIxPA_Dialogue_Dataset",
                         "train_data_path": "data_AmiciFamiglia_only_ground/train.json",
                         "dev_data_path": "data_AmiciFamiglia_only_ground/validation.json",
                     },
                     secrets=["HF_TOKEN"],
                     volumes=[{
                        "volume_type": "persistent_volume_claim",
                        "name": "volume-llmpa",
                        "mount_path": "/app/local_data",
                        "spec": { "size": "10Gi" }}]
					)
```

Please note that depending on the base model and the train data the amount of resources may vary. Adjust the size of the requested volume and be sure that sufficient amount of memory is provided. For the given example, for instance, around 40Gb of VRAM is required.
