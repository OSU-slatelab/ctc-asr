#!/bin/bash

#python main.py \
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --config-path="configs/" \
  --config-name="hc_ctc_all" \
  --node_rank 0
