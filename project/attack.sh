#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=1
current_datetime=$(date +"%Y%m%d_%H%M%S")

nohup python -u attack.py \
  --runname 'SP1000_10_N' \
  --datetime $current_datetime \
  > storage/logs/${current_datetime}_attack.log 2>&1 &