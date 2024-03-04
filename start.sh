#!/bin/bash

gpuopt=""
files=$(ls -l /dev/nvidia* 2> /dev/null | egrep -c '\n')
if [[ "$files" != "0" ]]
then
gpuopt="--nv"
fi

#this is a singularity problem only fixed recently
unset LD_LIBRARY_PATH
unset PYTHONPATH
sing=`which singularity`
unset PATH
cd

$sing run -B /eos -B /afs $gpuopt /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
