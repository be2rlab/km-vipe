#!/bin/bash

export SCENE_NAMES=(
    # Apartment_release_clean_seq134_M1292
    # Apartment_release_meal_skeleton_seq133_M1292
    Apartment_release_multiskeleton_party_seq102_M1292
)

export ARIA_EXISTING_CLASSES="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104"


length=${#SCENE_NAMES[@]}

for ((i=0; i<length; i++)); do
    scene_name=${SCENE_NAMES[i]}

    printf "Evaluating scene:   %s\n" "${scene_name}"

    python /eval/eval_semseg.py \
        --approach 'rtkm' \
        --semantic_info_path "/data/aria_Replica/${scene_name}/pcd/semseg_classes.json" \
        --existed_classes "${ARIA_EXISTING_CLASSES}" \
        --excluded_classes "0 8" \
        --pred_pc_path "/home/user/km-vipe/vipe_results/vipe/aria/${scene_name}" \
        --gt_pc_path "/data/aria_Replica/${scene_name}/pcd/gt.ply" \
        --output_path "/home/user/km-vipe/vipe_results/rtkm/aria" \
        --result_tag "${scene_name}" \
        --clip_prompts "There is {} in the scene" \
        --clip_name "ViT-B-16-quickgelu" \
        --clip_pretrained "openai" \
        --device "cuda:0" \
        --dataset "aria"
done


python /eval/compute_metrics.py \
    --results_dir "/home/user/km-vipe/vipe_results/rtkm/aria" \
    --excluded "-1 0 8" \
    --chunks '{
        "head": [8, 55, 1, 23, 36, 22, 6, 19, 15, 21, 25, 35, 75, 10, 7, 27, 52, 87, 26, 5, 91, 30, 2, 48, 92, 31, 9, 4, 24, 83, 13, 18, 78, 12, 62],
        "common": [43, 59, 20, 11, 17, 61, 28, 88, 40, 38, 80, 67, 41, 3, 64, 32, 16, 86, 58, 34, 79, 65, 72, 56, 29, 50, 47, 14, 74, 97, 45, 66, 100, 90],
        "tail": [69, 99, 33, 63, 81, 53, 84, 44, 49, 82, 54, 89, 46, 95, 42, 37, 73, 57, 98, 85, 39, 68, 76, 93, 70, 94, 77, 105, 51, 60, 71, 103, 101, 96, 102, 104]
    }'