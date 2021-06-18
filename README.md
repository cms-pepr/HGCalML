HGCalML
===============================================================================

Requirements
  * DeepJetCore 3.X (``https://github.com/DL4Jets/DeepJetCore``)
  * DeepJetCore 3.X container (or latest version in general)
  
For CERN, a script to start the latest container in interactive mode can be found here:

``/eos/home-j/jkiesele/singularity/run_deepjetcore3.sh``


Setup
===========

```
git clone  --recurse-submodules  https://github.com/cms-pepr/HGCalML
cd HGCalML
source env.sh #every time
./setup.sh #just once
```


When developing custom CUDA kernels
===========

The kernels are located in 
``modules/compiled``
The naming scheme is obvious and must be followed. Compile with make.



Converting the data from ntuples
===========

``convertFromSource.py -i <text file listing all training input files> -o <output dir> -c TrainData_window_onlytruth``

This data structure removes all noise and not correctly assigned truth showers until we have a better handle on the truth. Once we do, we can use ``TrainData_window`` which does not remove noise


Standard training and inference
===========
Go to the `Train` folder and then use the following command to start training. The file has code for running plots and more. That can be adapted according to needs.


```
cd Train

python3 june_format_example_nf_pca_double_coords.py /mnt/ceph/users/sqasim/Datasets/hgcal_kenneth_test_april_20_prop/dataCollection.djcdc /mnt/ceph/users/sqasim/trainings/training_folder
```

After training the model for a while, navigate to scripts directory and do the inference:

```
predict_hgcal.py /mnt/ceph/users/sqasim/trainings/training_folder/KERAS_check_model_last_save/ /mnt/ceph/users/sqasim/Datasets/hgcal_kenneth_test_april_20_prop/dataCollection.djcdc /mnt/ceph/users/sqasim/Datasets/hgcal_kenneth_test_april_20_prop/test_files.txt /mnt/ceph/users/sqasim/trainings/training_folder/inference
```


