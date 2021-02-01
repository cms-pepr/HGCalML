#!/bin/bash

TF_ROOT=/usr/local/lib/python3.6/dist-packages/tensorflow
CUDA_LINK=include/third_party/gpus

mkdir -p ${TF_ROOT}/${CUDA_LINK}
ln -sf /usr/local/cuda ${TF_ROOT}/${CUDA_LINK}/cuda
