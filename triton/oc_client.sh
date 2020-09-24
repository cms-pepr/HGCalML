#!/bin/bash

docker run -it --rm \
       --network oc-network \
       -v`pwd`/oc_client:/oc_client \
       nvcr.io/nvidia/tritonserver:20.08-py3-clientsdk \
       python /oc_client/triton_test.py -u ocserver:8001 -m hgcal_oc_reco
