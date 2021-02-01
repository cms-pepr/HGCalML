#!/bin/bash

if [ ! -e /dev/nvidia-uvm ]
then
echo "setting up nvidia devices in /dev/"

for i in `seq 0 7`
do
CUDA_VISIBLE_DEVICES=$i python -c "import torch;  print(torch.rand(2,3).cuda())" 
done

fi

path=`pwd` #/eos/home-j/jkiesele/singularity/triton/
image=/eos/home-j/jkiesele/singularity/triton/tritonserver_20.08-tfpepr-py3.sif
#this is a singularity problem only fixed recently
unset LD_LIBRARY_PATH
unset PYTHONPATH
sing=`which singularity`
unset PATH
cd 

$sing run -B /afs -B /eos -B $path:/oc_triton  --nv $image /oc_triton/cmssw_sing_runcommand.sh

#tritonserver --model-repository=/oc_triton/oc_models --backend-config=tensorflow,version=2 --log-verbose=1 --log-error=1 --log-info=1

