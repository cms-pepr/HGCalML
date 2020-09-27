#!/bin/bash


#wait for server

fifo_location="/dev/shm/${1}"
server=dockerbuild.cern.ch
port=8001

#wait for server to be available
check=`nc -vz ${server} 8001  2>&1 | grep "Connected to"`
while [[ ! $check ]]
do
sleep 1
echo $check
check=`nc -vz ${server} 8001  2>&1 | grep "Connected to"`
done


path=`pwd`
echo "connecting to triton server"
sing=`which singularity`
sys_rm=`which rm`
unset PATH
cd 

$sing run  \
       -B$path/oc_client:/oc_client \
       /eos/home-j/jkiesele/singularity/triton/tritonserver_20.08-py3-clientsdk.sif \
       python /oc_client/triton_test.py -u $server:$port  -m hgcal_oc_reco

       
