#!/usr/bin/env python3

import sys
import os

HGCALML=os.getenv("HGCALML")
if HGCALML is None:
    print('run with HGCALML environment sourced')
    exit()


if '-h' in sys.argv or '--help' in sys.argv:
    print('script to submit commands within the djc container to sbatch.\n')
    print('all commands are fully forwarded with one exception:')
    print('\n    ---d <workdir>    specifies a working directory can be specified that\n                      will contain the batch logs. It is created if \n                      it does not exist.\n')
    exit()

#can be used by others on FI
djcloc='/mnt/ceph/users/jkieseler/containers/deepjetcore3_latest.sif'

workdir=None

filtered_clo=[]
triggered=False
for clo in sys.argv:
    if clo == '---d':
        triggered=True
        continue
    if triggered:
        workdir=clo
        triggered=False
        continue
    filtered_clo.append(clo)

if workdir is None:
    print('please specify a batch working directory with ---d <workdir>')
    exit()

if os.path.isdir(workdir):
    var = input('Working directory exists, are you sure you want to continue, please type "yes/y"\n')
    var = var.lower()
    if not (var == 'yes' or var == 'y'):
        exit()
else:
    os.system('mkdir -p '+workdir)


filtered_clo = filtered_clo[1:] #remove
commands = " "
for clos in filtered_clo:
    commands += clos + " "

print(commands)

bscript_temp='''#!/bin/bash

#SBATCH  -p gpu --gres=gpu:1  --mincpus 4 -t 7-0 --constraint=a100-40gb

singularity  run  -B /mnt --nv {djcloc} /bin/bash -c "~/private/keytabd.sh & KTPID=$! ; cd {hgcalml}; source env.sh; cd - ;  {commands} ; kill $KTPID ; exit"

'''.format(djcloc=djcloc,
            hgcalml=HGCALML, 
            commands=commands )

with open(workdir+'/batchscript.sh','w') as f:
    f.write(bscript_temp)

os.system('cd '+workdir + '; pwd; module load slurm singularity; sbatch batchscript.sh')


