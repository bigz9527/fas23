#!/usr/bin/env bash

dataroot=$1

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --output_dir checkpoints --data-path $dataroot
