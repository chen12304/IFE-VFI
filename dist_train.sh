#!/usr/bin/env bash

WORLD_SIZE=$1
BATCH_SIZE=$2
DATA_PATH=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-1}
PORT=${PORT:-29600}
MASTER_ADDR=${MASTER_ADDR:-localhost}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --rdzv_backend=c10d\
    --rdzv_id=1\
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --world_size $WORLD_SIZE \
    --batch_size $BATCH_SIZE\
    --data_path $DATA_PATH