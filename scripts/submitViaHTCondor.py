#!/usr/bin/env python3
"""
Script to submit training scripts to Topas via HTCondor
"""

import sys
import os
import uuid
import subprocess

import re

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
    'n' : meta_option('n','TrainJob_'+UEXT),
    'f' : meta_option('f', '/work/friemer/hgcalml/HGCalML'), 
    'cpu': meta_option('cpu', '1'),
    'memory': meta_option('memory', '15 GB'),
    'disk': meta_option('disk', '8 GB'),
    'gpu': meta_option('gpu', '1'),
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
    print('\n    ---n <name> (opt) specifies a name for the scripts\n')
    print('\n    ---f <filepath> location of the HGCalML-Folder with necessary modules\n')
    print('\n    ---cpu <number> (opt) number of cpus to request default: 1\n')
    print('\n    ---memory <memory size> (opt) size of memory to request default: 15 GB\n')
    print('\n    ---disk <disk size> (opt) size of memory to request default 8 GB\n')
    print('\n    ---gpu <number> (opt) number of gpus to request default: 1\n')
    sys.exit()

if os.path.isdir(opts['n'].value):
    var = input(\
        'Working directory exists, are you sure you want to continue, please type "yes/y"\n')
    var = var.lower()
    if not var in ('yes', 'y'):
        sys.exit()
else:
    os.system('mkdir -p '+opts['n'].value)


filtered_clo = filtered_clo[1:] #remove
COMMANDS = " "
for clos in filtered_clo:
    COMMANDS += clos + " "

CWD = os.getcwd()


####################################################################################################
### Create Sub File #############################################################################
####################################################################################################

#Get absolute filepath and transfer entire folder if there is a .djcdc file also replace filepaths with filenames
inputfileslocations =''
NEWCOMMANDS = ''
for word in COMMANDS.split():
    if os.path.exists(word):
        if '.djcdc' in word:
            inputfileslocations +=  os.path.join(CWD, os.path.dirname(word))+ '/, '
        else:
            inputfileslocations += os.path.join(CWD, word) + ', '
        NEWCOMMANDS+=os.path.basename(word) + ' '
    else:
        NEWCOMMANDS+=word + ' '

#Create a tarball of the HGCalML folder
if os.path.isdir(opts['f'].value):
    os.system(f'''
            cd {opts['f'].value}
            cd ../
            tar -czf {opts['n'].value}/HGCalML.tar.gz {os.path.basename(opts['f'].value)}
            cd {CWD}''')
elif os.path.isfile(opts['f'].value) and opts['f'].value.endswith('.tar.gz'):
    os.system(f'''
            cp {opts['f'].value} {opts['n'].value}/HGCalML.tar.gz''')
else:
    raise Exception('Folder not found, please specify the correct path to the HGCalML-folder with ---f <filepath>')

#Setup the sub file
sub_temp=f'''#!/bin/bash
Universe = docker
docker_image = cernml4reco/deepjetcore4:abc9aee
accounting_group = cms.jet

requirements =(TARGET.CloudSite=="topas")
+RemoteJob = True
+RequestWalltime = 72 * 60 * 60

executable = {opts['n'].value+'_'+UEXT+'_run.sh'}

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files =  {inputfileslocations} {CWD+ '/' +opts['n'].value}/HGCalML.tar.gz, {CWD+ '/' +opts['n'].value+ '/' + opts['n'].value+'_'+UEXT+'_run.sh'}
transfer_output_files = .

output = {opts['n'].value+'_'+UEXT}.out
error = {opts['n'].value+'_'+UEXT}.err
log = {opts['n'].value+'_'+UEXT}.log

request_cpus = {opts['cpu'].value}
request_memory = {opts['memory'].value}
request_disk = {opts['disk'].value}
request_GPUs = {opts['gpu'].value}
'''

# Command to get WANDB_API_KEY
script_path = os.path.expanduser('~/private/wandb_api.sh')
command = f'''
echo "Checking for wandb API Key"
if [ -f {script_path} ]; then
   source {script_path}
   echo $WANDB_API_KEY
fi'''
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

with open(opts['n'].value+'/'+opts['n'].value+'_'+UEXT+'.sub','w', encoding='utf-8') as f:
    f.write(sub_temp)


####################################################################################################
### Create Run File #############################################################################
####################################################################################################

#Setup Run script
runscript_temp=f'''
#!/bin/bash
tar -xzf HGCalML.tar.gz

export HGCALML=$(readlink -f HGCalML)
export DEEPJETCORE_SUBPACKAGE=$HGCALML

export PATH=$HGCALML/scripts:$PATH
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
export LD_LIBRARY_PATH=$HGCALML/modules:$LD_LIBRARY_PATH
export LC_ALL=C.UTF-8 	# necessary for wandb
export LANG=C.UTF-8    # necessary for wandb

ls -l

{NEWCOMMANDS}

ls -l

rm HGCalML.tar.gz
rm -r HGCalML
rm *.djctd
rm *.djcdc

rm -r tmp
rm -r var
rm -r wandb

ls -l
'''
with open(opts['n'].value+'/'+opts['n'].value+ '_' +UEXT+'_run.sh','w', encoding='utf-8') as f:
    f.write(runscript_temp)



####################################################################################################
### Submit the Job #############################################################################
####################################################################################################

SUBMITCOMMAND = (
    'cd ' + opts['n'].value + '; pwd; condor_submit ' + opts['n'].value + '_' + UEXT + '.sub'
)
os.system(SUBMITCOMMAND)