#!/bin/bash

export SCENE_NAMES=(
    room0
    room1
    room2 
    office0
    office1
    office2
    office3
    office4
)

for scene_name in ${SCENE_NAMES[*]}
do
    printf "Running scene:   %s\n" "$scene_name"

    python run.py \
        pipeline=replica \
        streams=frame_dir_stream \
        streams.base_path=/data/Replica/${scene_name}/rgb
done