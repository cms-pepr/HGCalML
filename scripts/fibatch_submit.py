#!/usr/bin/env python3

import sys
import os
import uuid


HGCALML=os.getenv("HGCALML")
if HGCALML is None:
    print('run with HGCALML environment sourced')
    exit()


if '-h' in sys.argv or '--help' in sys.argv:
    print('script to submit commands within the djc container to sbatch.\n')
    print('all commands are fully forwarded with one exception:')
    print('\n    ---d <workdir>    specifies a working directory can be specified that\n                      will contain the batch logs. It is created if \n                      it does not exist.\n')
    print('\n    ---n <name> (opt) specifies a name for the batch script\n')
    print('\n    ---c <constraint> (opt) specifies a resource constraint, default a100\n')
    
    exit()

#can be used by others on FI
djcloc='/mnt/ceph/users/jkieseler/containers/deepjetcore3_latest.sif'

uext = str(uuid.uuid4())

workdir=None

filtered_clo=[]
triggered=False
triggeredname=False
triggeredconstraint=False
name="batchscript"
constraint="a100"
for clo in sys.argv:
    
    if clo == '---n':
        triggeredname = True
        continue
    if triggeredname:
        name = clo
        triggeredname=False
        continue
    
    if clo == '---d':
        triggered=True
        continue
    if triggered:
        workdir=clo
        triggered=False
        continue
    
    if clo == '---c':
        triggeredconstraint=True
        continue
    if triggeredconstraint:
        constraint=clo
        triggeredconstraint=False
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

CWD = os.getcwd()

bscript_temp='''#!/bin/bash

#SBATCH  -p gpu --gres=gpu:1  --mincpus 4 -t 3-0 --constraint={constraint}

nvidia-smi
singularity  run  -B /mnt --nv {djcloc} /bin/bash runscript_{uext}.sh

'''.format(djcloc=djcloc,
           uext=uext,
            workdir=workdir,
            constraint=constraint)

runscript_temp='''
~/private/keytabd.sh >/dev/null & 
KTPID=$!
cd {hgcalml}
source env.sh
cd {cwd}
{commands}
kill $KTPID
exit
'''.format(hgcalml=HGCALML, 
            cwd=CWD,
            commands=commands )


with open(workdir+'/'+name+'_'+uext+'.sh','w') as f:
    f.write(bscript_temp)
    
with open(workdir+'/runscript_'+uext+'.sh','w') as f:
    f.write(runscript_temp)
    
os.system('cd '+workdir + '; pwd; module load slurm singularity; unset PYTHONPATH ; sbatch '+name+'_'+uext+'.sh')


