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

#docker rm -f $CONTAINER_NAME 2>/dev/null

docker run -d $COMMON_ARG $DOCKER_IMG sleep infinity

#docker exec -w /workspace $CONTAINER_NAME ls -al | tee -a fasop_run.log

#run model for TP=1 2 4
for TP in 1 2 4; do
	docker exec -w /workspace/aicomp/compiler_fx $CONTAINER_NAME sh -c "torchrun --nproc_per_node=$TP --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=29500  llama_2d_local4fasop_prof.py $HUGGINGFACE_TOKEN | tee /workspace/tdpp/log2/tp_output/tp$TP_output.out"

#get mean numbers from log
	docker exec -w /workspace/tdpp/Megatron-LM $CONTAINER_NAME sh -c "python3 _06_get_layer_time.py | grep mean | awk '{print $2}' > tp$TP_fasop.txt"
done

{
	read layer_mean_1
	read emb_mean_1
	read post_mean_1
} < tp1_fasop.txt

{
	read layer_mean_2
	read emb_mean_2
	read post_mean_2
} < tp2_fasop.txt

{
	read layer_mean_4
	read emb_mean_4
	read post_mean_4
} < tp4_fasop.txt

#generate npy file
SURFIX=$(date +%Y-%m-%d)
docker exec -w /workspace/tdpp/FASOP/known_cost $CONTAINER_NAME sh -c "python3 profile.py --model_name gpt2XL --gpu_type A40 --transformer_type de --add_name $SURFIX --decoder_embedding_time $emb_mean_1 $emb_mean_2 $emb_mean_4 --decoder_time $layer_mean_1 $layer_mean_2 $layer_mean_4 --decoder_post_process_time $post_mean_1 $post_mean_2 $post_mean_4 --de_layer_num 16"

docker cp $CONTAINER_NAME:/workspace/tdpp/FASOP/known_cost/gpt2XL_A40_?_$SURFIX.npy ./
#docker stop $CONTAINER_NAME
#docker rm $CONTAINER_NAME
