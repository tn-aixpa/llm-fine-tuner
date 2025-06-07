# Train LLM Adapter

It is possible to train LLM adapter using the docker container. 

## Build container

The container spec is present in ``src`` folder. 

```bash
docker build --no-cache --rm -t ghcr.io/tn-aixpa/llm-fine-tuner .
```

Change the name of the container at your will.

The data produced by the execution will be stored at the following path: ``/app/local_data``. Mount the volume accordingly, 

## Run container

To run the container, the execution arguments should be provided. Additionally, if HuggingFace and/or Weight-and-Biases are required,
they should be provided as environment entries:

```bash
docker run -d --name aixpa-training-0 --gpus "device=0"  -e HF_TOKEN=<TOKEN_VALUE> llm-fine-tuner --model_id=Llama-3.1-8B ...
```

## Run container within the platform

To run the container within the platform, it is necessary to define the corresponding container function:

```python
import  digitalhub as dh

project = dh.get_or_create_project("llmpa")

func = project.new_function(
    name="train-llm", 
    kind="container", 
    image="ghcr.io/tn-aixpa/llm-fine-tuner"
)
```

Once defined, create the corresponding run with the necessary parameters. To log the model, set the ``log_model`` flag and specify the model name (``model_name``).

It is important to note that for the execution a volume should be created as the space requirements exceed the default available space.

The huggingface token and wandb key should be passed as project secrets or as env variables (not recommended). Create the corresponding secrets (``HF_TOKEN`` and ``WANDB_KEY``) in the project configuration.

```python

train_run = func.run(action="job",
                     profile="1xa100",
                     args=[
                         "--model_id=meta-llama/Llama-3.1-8B",
                         "--log_model",
                         "--model_name=llmpa",
                         "--hf_dataset_name=LanD-FBK/AIxPA_Dialogue_Dataset",
                         "--train_data_path=data_AmiciFamiglia_only_ground/train.json",
                         "--dev_data_path=data_AmiciFamiglia_only_ground/validation.json"
                     ],
                     secrets=["HF_TOKEN"],
                     volumes=[{
                        "volume_type": "persistent_volume_claim",
                        "name": "volume-llmpa",
                        "mount_path": "/app/local_data",
                        "spec": { "size": "10Gi" }}]
					)
```

Once complete, the model adapter is automatically logged in the project under the specified name (``llmpa`` in this case).

