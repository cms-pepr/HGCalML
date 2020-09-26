#!/bin/bash

if [ ! -e /dev/nvidia-uvm ]
then
echo "setting up nvidia devices in /dev/"

for i in `seq 0 7`
do
CUDA_VISIBLE_DEVICES=$i python -c "import torch;  print(torch.rand(2,3).cuda())" 
done

fi

path=/eos/home-j/jkiesele/singularity/triton/
image=/eos/home-j/jkiesele/singularity/triton/tritonserver_20.08-tfpepr-py3.sif
#this is a singularity problem only fixed recently
unset LD_LIBRARY_PATH
unset PYTHONPATH
sing=`which singularity`
unset PATH
cd 
echo "If you see the following error: \"container creation failed: mount /proc/self/fd/10->/var/singularity/mnt/session/rootfs error ...\" please just try again"
$sing run -B /home  -B /eos -B /afs --bind /etc/krb5.conf:/etc/krb5.conf --bind /proc/fs/openafs/afs_ioctl:/proc/fs/openafs/afs_ioctl --bind /usr/vice/etc:/usr/vice/etc  --nv $image  tritonserver --model-repository=$path/oc_models --backend-config=tensorflow,version=2 --log-verbose=1 --log-error=1 --log-info=1
