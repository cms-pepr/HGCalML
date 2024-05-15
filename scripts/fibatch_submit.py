#!/usr/bin/env python3
"""
Script to submit batch jobs to the FI cluster.
"""

import sys
import os
import uuid


HGCALML=os.getenv("HGCALML")
if HGCALML is None:
    print('run with HGCALML environment sourced')
    sys.exit()


if '-h' in sys.argv or '--help' in sys.argv:
    print('script to submit commands within the djc container to sbatch.\n')
    print('all commands are fully forwarded with one exception:')
    print('\n    ---d <workdir>    specifies a working directory that can be specified '
      'that will contain the batch logs. It is created if it does not exist.\n')
    print('\n    ---n <name> (opt) specifies a name for the batch script\n')
    print('\n    ---c <constraint> (opt) specifies a resource constraint, default a100\n')
    sys.exit()

# can be used by others on FI
DJCLOC="/mnt/home/pzehetner/containers/deepjetcore4_5ef28a3.sif"
UEXT = str(uuid.uuid4())

WORKDIR=None

filtered_clo=[]
TRIGGERED=False
TRIGGERED_NAME=False
TRIGGERED_CONSTRAINT=False
TRIGGERED_TIME=False
TRIGGERED_GPU=False
NAME="batchscript"
# constraint="a100"
CONSTRAINT="a100-80gb"
#CONSTRAINT="a100"
TIME="7-0"
GPUS="1"

for clo in sys.argv:

    if clo == '---n':
        TRIGGERED_NAME = True
        continue
    if TRIGGERED_NAME:
        NAME = clo
        TRIGGERED_NAME=False
        continue

    if clo == '---g':
        TRIGGERED_GPU = True
        continue
    if TRIGGERED_GPU:
        GPUS = clo
        TRIGGERED_GPU=False
        continue

    if clo == '---d':
        TRIGGERED=True
        continue
    if TRIGGERED:
        WORKDIR=clo
        TRIGGERED=False
        continue

    if clo == '---c':
        TRIGGERED_CONSTRAINT=True
        continue
    if TRIGGERED_CONSTRAINT:
        CONSTRAINT=clo
        TRIGGERED_CONSTRAINT=False
        continue
    
    if clo == '---t':
        TRIGGERED_TIME=True
        continue
    if TRIGGERED_TIME:
        TIME=clo
        TRIGGERED_TIME=False
        continue

    filtered_clo.append(clo)

if WORKDIR is None:
    print('please specify a batch working directory with ---d <workdir>')
    sys.exit()

if os.path.isdir(WORKDIR):
    var = input(\
        'Working directory exists, are you sure you want to continue, please type "yes/y"\n')
    var = var.lower()
    if not var in ('yes', 'y'):
        sys.exit()
else:
    os.system('mkdir -p '+WORKDIR)


filtered_clo = filtered_clo[1:] #remove
COMMANDS = " "
for clos in filtered_clo:
    COMMANDS += clos + " "

CWD = os.getcwd()

bscript_temp=f'''#!/bin/bash

#SBATCH  -p gpu --gres=gpu:{GPUS}  --mincpus 4 -t 7-0 --constraint={CONSTRAINT}

nvidia-smi
singularity  run  -B /mnt --nv {DJCLOC} /bin/bash runscript_{UEXT}.sh

'''

runscript_temp=f'''
~/private/keytabd.sh >/dev/null &
KTPID=$!
cd {HGCALML}
source env.sh
cd {CWD}
{COMMANDS}
kill $KTPID
exit
'''

with open(WORKDIR+'/'+NAME+'_'+UEXT+'.sh','w', encoding='utf-8') as f:
    f.write(bscript_temp)

with open(WORKDIR+'/runscript_'+UEXT+'.sh','w', encoding='utf-8') as f:
    f.write(runscript_temp)

COMMAND = (
    'cd ' + WORKDIR + '; pwd; module load slurm singularity; unset PYTHONPATH ; '
    'sbatch ' + NAME + '_' + UEXT + '.sh'
)
os.system(COMMAND)
