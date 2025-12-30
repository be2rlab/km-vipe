#!/bin/bash
set -e

# Run ONNX export
python scripts/unidepth_onnx.py --backbone vitl --shape 336 602 --with-camera

# Build TensorRT engine
trtexec --onnx=weights/unidepthv2_c.onnx --fp16 --saveEngine=weights/unidepthv2-l-c-336-602.engine
