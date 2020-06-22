#!/usr/bin/env bash

exe_path="/home/shuocheng/fusibile/fusibile"
root_path="/new1/shuocheng/dtu_results"
target_path="/new1/shuocheng/dtu_points"



declare -a arr=(1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118)

for i in ${arr[@]}; do
    scene_path="$root_path/scan$i"
    CUDA_VISIBLE_DEVICES=0 python depthfusion.py --dense_folder $scene_path --fusibile_exe_path $exe_path --prob_threshold 0.6 --disp_threshold 0.25 --num_consistent 3
done

python utils/collect_pointclouds.py --root_dir $root_path --target_dir $target_path --dataset "dtu"