#!/bin/bash
set -e

trtexec --onnx=/models/xl0_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=/workspace/perception/algorithms/models/engines/xl0_encoder.engine

trtexec --onnx=/models/xl0_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:128x2x2,point_labels:128x2 --maxShapes=point_coords:128x2x2,point_labels:128x2 --fp16 --saveEngine=/workspace/perception/algorithms/models/engines/xl0_decoder.engine

# trtexec --onnx=/models/l0_encoder.onnx --minShapes=input_image:1x3x512x512 --optShapes=input_image:4x3x512x512 --maxShapes=input_image:4x3x512x512 --saveEngine=/workspace/perception/algorithms/models/engines/l0_encoder.engine

# trtexec --onnx=/models/l0_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:128x2x2,point_labels:128x2 --maxShapes=point_coords:128x2x2,point_labels:128x2 --fp16 --saveEngine=/workspace/perception/algorithms/models/engines/l0_decoder.engine

python3 /workspace/perception/algorithms/models/convert_object_detector.py
cp /workspace/perception/algorithms/models/weight/ObjectAwareModel.engine /workspace/perception/algorithms/models/engines/ObjectAwareModel.engine

exec "$@"
