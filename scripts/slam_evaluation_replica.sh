#!/bin/bash

export ROOT_DIR=/home/user/km-vipe
export GT_FOLDER=/data/replica
export RESULTS_FOLDER=$ROOT_DIR/vipe_results
export SCENE_NAMES=(
    room0
    # room1
    # room2 
    # office0
    # office1
    # office2
    # office3
    # office4
)

for SCENE_NAME in ${SCENE_NAMES[*]}
do
    printf "Running scene:   %s\n" "$SCENE_NAME"

    # python3 $ROOT_DIR/scripts/create_video.py \
    #     --dataset "replica" \
    #     --scene_name=$SCENE_NAME \
    #     --input_dir=$GT_FOLDER \
    #     --output_dir=$RESULTS_FOLDER/videos/ \

    python3 $ROOT_DIR/run.py \
        pipeline=replica \
        streams=raw_mp4_stream \
        pipeline.output.save_artifacts=true \
        streams.base_path=$RESULTS_FOLDER/videos/$SCENE_NAME.mp4 \
        pipeline.output.path=$RESULTS_FOLDER \
        pipeline.slam.dataset.sequence_name=$SCENE_NAME \
        pipeline.slam.keyframe_depth=unidepth-l \

    python $ROOT_DIR/scripts/rmse_evaluation.py \
        --dataset "replica" \
        --gt_folder "$GT_FOLDER" \
        --results_folder "$RESULTS_FOLDER" \
        --scene_name "$SCENE_NAME" \
        --metrics_path "$RESULTS_FOLDER/metrics.csv" \
        --plot \
        --save_plot "$RESULTS_FOLDER/rmse" \

done 