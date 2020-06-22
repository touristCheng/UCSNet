#!/usr/bin/env bash

root_path="/cephfs/shuocheng/mvs_training/dtu"

root_path="/new1/shuocheng/dtu/mvs_training/dtu"

save_path="./training_$(date +"%F-%T")"
num_gpus=$1
batch=2

mkdir -p $save_path

python -m torch.distributed.launch --nproc_per_node=$num_gpus train.py --root_path=$root_path --save_path $save_path \
          --batch_size $batch --epochs 60 --lr 0.0016 --lr_idx "20,30,40,50:0.625" --loss_weights "0.5,1.0,2.0" \
           --net_configs "64,32,8" --num_views 2 --lamb 1.5 --sync_bn | tee -a $save_path/log.txt
