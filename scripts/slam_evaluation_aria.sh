#!/bin/bash

export ROOT_DIR=/home/user/km-vipe
export GT_FOLDER=/data/aria_Replica/
export RESULTS_FOLDER=$ROOT_DIR/aria_results
export SCENE_NAMES=(
Apartment_release_clean_seq134_M1292            
# Apartment_release_meal_skeleton_seq133_M1292
# Apartment_release_multiskeleton_party_seq102_M1292 
)

for SCENE_NAME in ${SCENE_NAMES[*]}
do
    printf "Running scene:   %s\n" "$SCENE_NAME"

    # python3 $ROOT_DIR/scripts/create_video.py \
    #     --dataset=aria \
    #     --scene_name=$SCENE_NAME \
    #     --input_dir=$GT_FOLDER \
    #     --output_dir=$RESULTS_FOLDER/videos/ \

    python3 $ROOT_DIR/run.py \
        pipeline=default \
        streams=frame_dir_stream \
        streams.base_path=$GT_FOLDER/$SCENE_NAME/rgb \
        streams.scene_name=$SCENE_NAME \
        pipeline.output.save_artifacts=true \
        pipeline.output.path=$RESULTS_FOLDER \
        # pipeline.slam.dataset.sequence_name=$SCENE_NAME \
        # pipeline.slam.keyframe_depth=dataset \

    # python $ROOT_DIR/scripts/rmse_evaluation.py \
    #     --dataset "aria" \
    #     --gt_folder "$GT_FOLDER" \
    #     --results_folder "$RESULTS_FOLDER" \
    #     --scene_name "$SCENE_NAME" \
    #     --metrics_path "$RESULTS_FOLDER/metrics.csv" \
    #     --plot \
    #     --save_plot "$RESULTS_FOLDER/rmse" \

done 