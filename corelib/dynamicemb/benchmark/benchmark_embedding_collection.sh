#!/bin/bash

USE_INDEX_DEDUP=${1:-True}
USE_DYNAMIC_EMBEDDING=${2:-True}
BATCH_SIZE=${3:-65536}

echo "Use Index Dedup: $USE_INDEX_DEDUP"
echo "Use Dynamic Embedding: $USE_DYNAMIC_EMBEDDING"
echo "Batch Size: $BATCH_SIZE"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  ./benchmark/benchmark_embedding_collection.py --use_index_dedup $USE_INDEX_DEDUP --use_dynamic_embedding $USE_DYNAMIC_EMBEDDING --batch_size $BATCH_SIZE
