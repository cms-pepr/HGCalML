#!/bin/bash

docker network create oc-network

docker run --rm -it  -p8000:8000 -p8001:8001 -p8002:8002 \
       -v`pwd`/oc_models:/oc_models \
       --network oc-network \
       --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
       --name ocserver \
       jkiesele/tritonserver:20.08-tfpepr-py3 \
       tritonserver --model-repository=/oc_models --backend-config=tensorflow,version=2 --log-verbose=1 --log-error=1 --log-info=1
