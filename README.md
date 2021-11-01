HGCalML
===============================================================================

Requirements
  * DeepJetCore 3.X (``https://github.com/DL4Jets/DeepJetCore``)
  * DeepJetCore 3.X container (or latest version in general)
  
For CERN (or any machine with cvmfs mounted), a script to start the latest container use this script:
```
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
```

The package follows the structure and logic of all DeepJetCore subpackages (also the example in DeepJetCore). So as a fresh starting point, it can be a good idea to follow the DeepJetCore example first.

Setup
===========

```
git clone  --recurse-submodules  https://github.com/cms-pepr/HGCalML
cd HGCalML
source env.sh #every time
./setup.sh #just once, compiles custom kernels
```


When developing custom CUDA kernels
===========

The kernels are located in 
``modules/compiled``
The naming scheme should be obvious and must be followed. Compile with make.



Converting the data from ntuples
===========

``convertFromSource.py -i <text file listing all training input files> -o <output dir> -c TrainData_NanoML``
The conversion rule itself is located here:
``modules/datastructures/TrainData_NanoML.py``

The training files (see next section) usually also contain a comment in the beginning pointing to the latest data set at CERN and flatiron.

Standard training and inference
===========
Go to the `Train` folder and then use the following command to start training. The file has code for running plots and more. That can be adapted according to needs.


```
cd Train

python3 simplified_trainer_example.py /mnt/ceph/users/sqasim/Datasets/hgcal_kenneth_test_april_20_prop/dataCollection.djcdc /mnt/ceph/users/sqasim/trainings/training_folder
```

After training the model for a while, navigate to scripts directory and do the inference. Please note that this is *different* from the standard DeepJetCore procedure.

```
predict_hgcal.py /mnt/ceph/users/sqasim/trainings/training_folder/KERAS_check_model_last_save/ /mnt/ceph/users/sqasim/trainings/training_folder/valsamples.djcdc /mnt/ceph/users/sqasim/Datasets/hgcal_kenneth_test_april_20_prop /mnt/ceph/users/sqasim/trainings/training_folder/inference
```


