#!/bin/bash
set -e

# Run ONNX export
python scripts/unidepth_onnx.py --backbone vitb --shape 672 1190

# Build TensorRT engine
trtexec --onnx=weights/unidepthv2.onnx --fp16 --saveEngine=weights/unidepthv2-672-1190.engine
