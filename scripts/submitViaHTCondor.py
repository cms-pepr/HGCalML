#!/usr/bin/env python3
"""
Script to submit training scripts to Topas via HTCondor
"""

import sys
import os
import uuid
import subprocess

UEXT = str(uuid.uuid4())


class meta_option(object):
    def __init__(self, id, default_val = None) -> None:
        self.id = id
        self.triggered = False
        self.value = default_val

    def check(self, clo): 
        if self.triggered:
            self.value = clo
            self.triggered = False
            return True
        if clo == '---'+self.id:
            self.triggered = True
            return True 
        return False
    
    def valid(self):
        return self.value is not None
    
    def __str__(self) -> str:
        return self.id + ': '+self.value
        

opts = {
    'd' : meta_option('d'),
    't' : meta_option('t', '/work/friemer/hgcalml/trainingdata_split/'),
    'm' : meta_option('m', '/work/friemer/hgcalml/HGCalML/Train/paper_trainer_noRSU.py'),
    'h' : meta_option('h', 'HGCalML'), 
    'n' : meta_option('n','TrainJob'),
    'cpu': meta_option('cpu', '1'),
    'memory': meta_option('memory', '15 GB'),
    'disk': meta_option('disk', '8 GB'),
}

filtered_clo=[]

for clo in sys.argv:
    next = False
    for _,o in opts.items():
        if o.check(clo):
            next = True
            break
    if next:
        continue
    filtered_clo.append(clo)

all_valid = True
for _,o in opts.items():
    all_valid = all_valid and o.valid()

if '-h' in sys.argv or '--help' in sys.argv or (not all_valid):
    print('script to submit commands within the  container to sbatch.\n')
    print('all commands are fully forwarded with one exception:')
    print('\n    ---d <workdir>    specifies a working directory that can be specified '
      'that will contain the files. It is created if it does not exist.\n')
    print('\n    ---t <filepath> specifies the location of the training-data to feed the model\n')
    print('\n    ---m <filepath> specifies which model to execute\n')
    print('\n    ---h <filepath> location of HGCalML-Folder with necessary modules\n')
    print('\n    ---n <name> (opt) specifies a name for the scripts\n')
    print('\n    ---cpu <number> (opt) number of cpus to request default: 1\n')
    print('\n    ---memory <memory size> (opt) size of memory to request default: 15 GB\n')
    print('\n    ---disk <disk size> (opt) size of memory to request default 8 GB\n')
    sys.exit()

if os.path.isdir(opts['d'].value):
    var = input(\
        'Working directory exists, are you sure you want to continue, please type "yes/y"\n')
    var = var.lower()
    if not var in ('yes', 'y'):
        sys.exit()
else:
    os.system('mkdir -p '+opts['d'].value)


filtered_clo = filtered_clo[1:] #remove
COMMANDS = " "
for clos in filtered_clo:
    COMMANDS += clos + " "

CWD = os.getcwd()

#Create a tarball of the HGCalML folder
os.system(f'''tar -czf {opts['d'].value}/HGCalML.tar.gz {opts['h'].value}''')

#Setup the sub file
sub_temp=f'''#!/bin/bash
Universe = docker
docker_image = cernml4reco/deepjetcore4:5ef28a3
accounting_group = cms.jet

requirements =(TARGET.CloudSite=="topas")
+RemoteJob = True
+RequestWalltime = 72 * 60 * 60

executable = {opts['n'].value+UEXT+'_run.sh'}

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files =  {opts['m'].value}, {opts['t'].value}, {CWD+ '/' +opts['d'].value}/HGCalML.tar.gz, {CWD+ '/' +opts['d'].value+ '/' + opts['n'].value+UEXT+'_run.sh'}
#transfer_output_files = {opts['n'].value+'_'+UEXT}

output = {opts['n'].value+'_'+UEXT}.out
error = {opts['n'].value+'_'+UEXT}.err
log = {opts['n'].value+'_'+UEXT}.log

request_cpus = {opts['cpu'].value}
request_memory = {opts['memory'].value}
request_disk = {opts['disk'].value}
'''

## Path to your shell script
script_path = os.path.expanduser('~/private/wandb_api.sh')

# Command to source the shell script and print the environment variable
command = f'''
echo "Checking for wandb API Key"
if [ -f {script_path} ]; then
   source {script_path}
   echo $WANDB_API_KEY
fi'''

# Execute the command and capture the output
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, executable='/bin/bash')

# Extract the WANDB_API_KEY from the command output
API_Key = result.stdout.strip().split('\n')[-1]
if API_Key is not None:
    sub_temp += f'''
environment = "WANDB_API_KEY={API_Key}"
'''
else:
    print('No WANDB_API_KEY found, please set it in your environment or in ~/private/wandb_api.sh')

#End of the sub file
sub_temp += f'''
queue
'''

with open(opts['d'].value+'/'+opts['n'].value+'_'+UEXT+'.sub','w', encoding='utf-8') as f:
    f.write(sub_temp)

#Setup Run script
runscript_temp=f'''
#!/bin/bash
tar -xzf HGCalML.tar.gz

export HGCALML=$(readlink -f HGCalML)
echo $HGCALML

export DEEPJETCORE_SUBPACKAGE=$HGCALML

export PATH=$HGCALML/scripts:$PATH
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
export LD_LIBRARY_PATH=$HGCALML/modules:$LD_LIBRARY_PATH
export LC_ALL=C.UTF-8 	# necessary for wandb
export LANG=C.UTF-8    # necessary for wandb

ls -l

python3 {os.path.basename(opts['m'].value)} dataCollection._n.djcdc {opts['n'].value+'_'+UEXT}
'''
with open(opts['d'].value+'/'+opts['n'].value+UEXT+'_run.sh','w', encoding='utf-8') as f:
    f.write(runscript_temp)

#Submit the job
COMMAND = (
    'cd ' + opts['d'].value + '; pwd; condor_submit ' + opts['n'].value + '_' + UEXT + '.sub'
)
os.system(COMMAND)