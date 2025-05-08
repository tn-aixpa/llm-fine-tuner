# Run the training

Before building the docker, in `run_training.sh` add:

- Add your HF_TOKEN and WANDB_KEY tokens (the key need to have the __permission to download__  Llama-3.1-8B-Instruct model. Authorization can be asked at this link: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct )
- Change The model and dataset paths (if different) and the output names
- Change the Training parameters as needed (eg with an appropriate batch size for your gpu)

To run the training

```
docker build  --rm -t aixpa-training .
```

```
docker run -d --name aixpa-training-0 —gpus "device=o" -v "$PWD":/code --shm-size=48gb aixpa-training
```

To check the progression status:
```
docker logs aixpa-training-0
```

> [!NOTE] 
> After running the docker it will show __the logs will not show any update for some minutes__ while it is downloading llama 3.1 model from huggingface
> To check it the model is downloading, it is saved within the docker in `/root/.cache/huggingface` (the script needs to download about 15GB)

# Models

All the checkpoint are saved in the `checkpoints` folder. The best one is saved in `weights` folder (unless renamed in `run_training.sh`

