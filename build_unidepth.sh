#!/bin/bash
set -e

# Run ONNX export
python unidepth_onnx.py --backbone vitb --shape 672 1190

# Build TensorRT engine
trtexec --onnx=unidepthv2.onnx --fp16 --saveEngine=unidepthv2-672-1190.engine
