#!/bin/bash

if [[ ! $1 ]]
then
echo "please specify a pipe name"
exit -1
fi

fifo_location="/dev/shm/${1}"
server=localhost:8001

singularity run  \
       -B`pwd`/oc_client:/oc_client \
       docker://nvcr.io/nvidia/tritonserver:20.08-py3-clientsdk \
       python /oc_client/triton_forward_client.py -u $server -f $fifo_location -m hgcal_oc_reco

       
rm -f $fifo_location "${fifo_location}_pred"