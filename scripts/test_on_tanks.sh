#!/usr/bin/env bash


#save_path="/new1/shuocheng/tanks_results"
#test_list="./dataloader/datalist/tanks/test.txt"
#root_path="/new1/shuocheng/tankandtemples"

save_path="/cephfs/shuocheng/tanks_results"
test_list="./dataloader/datalist/tanks/test.txt"
root_path="/cephfs/shuocheng/tankandtemples/intermediate"

CUDA_VISIBLE_DEVICES=0 python test.py --root_path $root_path --test_list $test_list --save_path $save_path --max_h 1080 --max_w 1920
