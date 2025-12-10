#!/bin/bash

export SCENE_NAMES=(office0) #office0 office1 office2 office3 office4 room0 room1 room2)
export SCENE_NAMES_UNDERSCORE=(office_0) #office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2)

export REPLICA_EXISTING_CLASSES="0 3 7 8 10 11 12 13 14 15 16 17 18 19 20 22 23 26 29 31 34 35 37 40 44 47 52 54 56 59 60 61 62 63 64 65 70 71 76 78 79 80 82 83 87 88 91 92 93 95 97 98"


length=${#SCENE_NAMES[@]}

for ((i=0; i<length; i++)); do
    scene_name=${SCENE_NAMES[i]}
    scene_name_underscore=${SCENE_NAMES_UNDERSCORE[i]}

    printf "Evaluating scene:   %s\n" "${scene_name}"

    python /eval/eval_semseg.py \
        --approach 'rtkm' \
        --semantic_info_path "/data/gt/replica/${scene_name_underscore}/habitat/info_semantic.json" \
        --existed_classes "${REPLICA_EXISTING_CLASSES}" \
        --excluded_classes "0" \
        --pred_pc_path "/home/user/km-vipe/vipe_results/vipe/v3/${scene_name}" \
        --gt_pc_path "/data/gt/replica/${scene_name_underscore}/habitat/mesh_semantic.ply" \
        --output_path "/home/user/km-vipe/vipe_results/rtkm/replica/v3" \
        --result_tag "${scene_name}" \
        --clip_prompts "There is {} in the scene" \
        --clip_name "ViT-B-16-quickgelu" \
        --clip_pretrained "openai" \
        --device "cuda:0"
done


python /eval/compute_metrics.py \
    --results_dir "/home/user/km-vipe/vipe_results/rtkm/replica/v3" \
    --excluded "-1 0" \
    --chunks '{
        "head": [93, 31, 40, 20, 12, 76, 80, 98, 97, 47, 37, 61, 8, 87, 18, 60, 11],
        "common": [88, 29, 10, 92, 7, 78, 59, 44, 34, 26, 54, 71, 91, 63, 3, 64, 52],
        "tail": [62, 56, 35, 95, 13, 15, 22, 70, 83, 17, 82, 65, 14, 19, 16, 23, 79]
    }'