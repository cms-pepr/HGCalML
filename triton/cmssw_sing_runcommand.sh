#!/bin/bash


export CUDA_VISIBLE_DEVICES=7
#unset CUDA_VISIBLE_DEVICES

tritonserver --model-repository=/oc_triton/oc_models --backend-config=tensorflow,version=2 --log-verbose=1 --log-error=1 --log-info=1
