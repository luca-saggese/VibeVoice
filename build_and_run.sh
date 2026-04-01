docker build . -t vibevoice
docker run --rm -it --gpus all -p 8085:3000 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /data/huggingface:/huggingface vibevoice