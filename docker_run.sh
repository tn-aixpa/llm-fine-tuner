docker run -d --name aixpa-training-0 --gpus "device=0" -v "$PWD":/code --rm -ti --shm-size=48gb aixpa-training
