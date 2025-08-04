#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <master_addr> <huggingface_token>"
    exit 1
fi

#common args
DOCKER_IMG=swsok/fasop_docker:pytorch_2.5.0-cuda12.4-cudnn9-devel
CONTAINER_NAME="fasop_docker"
COMMON_ARG="--gpus all --name $CONTAINER_NAME --ipc=host --network=host"
MASTER_ADDR=$1
HUGGINGFACE_TOKEN=$2

#docker run --gpus all -i -t --name cmh_2.3.1 --ipc=host --network=host swsok/fasop_docker:pytorch_2.5.0-cuda12.4-cudnn9-devel /bin/bash

docker rm -f $CONTAINER_NAME 2>/dev/null

docker run -d $COMMON_ARG $DOCKER_IMG sleep infinity

#docker exec -w /workspace $CONTAINER_NAME ls -al | tee -a fasop_run.log

#run TP=1
docker exec -w /workspace/aicomp/compiler_fx $CONTAINER_NAME sh -c "torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=29500  llama_2d_local4fasop_prof.py $HUGGINGFACE_TOKEN | tee /workspace/tdpp/log2/tp_output/tp1_output.out"

#get mean numbers from log
docker exec -w /workspace/tdpp/Megatron-LM $CONTAINER_NAME python3 _06_get_layer_time.py | tee tp1_fasop.txt


#docker stop $CONTAINER_NAME
#docker rm $CONTAINER_NAME
