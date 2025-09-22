#!/bin/bash

export ROOT_DIR=/home/user/km-vipe
export GT_FOLDER=/data/tum
export RESULTS_FOLDER=$ROOT_DIR/tum_results
export CUDA_VISIBLE_DEVICES=1
export SCENE_NAMES=(
    rgbd_dataset_freiburg1_desk
    # rgbd_dataset_freiburg3_sitting_xyz
    # rgbd_dataset_freiburg3_sitting_halfsphere
    # rgbd_dataset_freiburg3_sitting_static
    # rgbd_dataset_freiburg3_sitting_rpy
    # rgbd_dataset_freiburg3_walking_xyz
    # rgbd_dataset_freiburg3_walking_halfsphere
    # rgbd_dataset_freiburg3_walking_static
    # rgbd_dataset_freiburg3_walking_rpy
)


for SCENE_NAME in ${SCENE_NAMES[*]}
do
    printf "Running scene:   %s\n" "$SCENE_NAME"

    python3 $ROOT_DIR/scripts/create_video.py \
        --dataset=tum \
        --scene_name=$SCENE_NAME \
        --input_dir=$GT_FOLDER \
        --output_dir=$RESULTS_FOLDER/videos/ \

    python3 $ROOT_DIR/run.py \
        pipeline=default \
        streams=raw_mp4_stream \
        pipeline.output.save_artifacts=true \
        streams.base_path=$RESULTS_FOLDER/videos/$SCENE_NAME.mp4 \
        pipeline.output.path=$RESULTS_FOLDER \
        # pipeline.slam.dataset.sequence_name=$SCENE_NAME \
        # pipeline.slam.keyframe_depth=unidepth-l \

    python $ROOT_DIR/scripts/rmse_evaluation.py \
        --dataset "tum" \
        --gt_folder "$GT_FOLDER" \
        --results_folder "$RESULTS_FOLDER" \
        --scene_name "$SCENE_NAME" \
        --metrics_path "$RESULTS_FOLDER/metrics.csv" \
        --plot \
        --save_plot "$RESULTS_FOLDER/rmse" \

done 