#!/bin/bash

if [[ ! $1 ]]
then
echo "please specify a pipe name"
exit -1
fi

#wait for server

fifo_location="/dev/shm/${1}"
server=dockerbuild.cern.ch
port=8001

#wait for server to be available
check=`nc -vz ${server} 8001  2>&1 | grep "Connected to"`
while [[ ! $check ]]
do
sleep 1
check=`nc -vz ${server} 8001  2>&1 | grep "Connected to"`
done

#clean pipes


sys_rm=`which rm`

function finish {
$sys_rm -f $fifo_location "${fifo_location}_pred"
}
finish

trap finish EXIT SIGHUP SIGKILL SIGTERM

echo "server found... connecting to triton"
sing=`which singularity`
unset PATH
cd 

$sing run  \
       -B/eos/home-j/jkiesele/singularity/triton/oc_client:/oc_client \
       /eos/home-j/jkiesele/singularity/triton/tritonserver_20.08-py3-clientsdk.sif \
       python /oc_client/triton_forward_client.py -u $server:$port -f $fifo_location -m hgcal_oc_reco

finish
       
