#!/usr/bin/env bash

datalist=$1
dataroot=$2
output=$3

python3 -u main.py --eval True --dist-eval False --batch-size 256 --resume checkpoints/best.pth  --datalist-path $datalist --data-path $dataroot  --score_out_path $output
