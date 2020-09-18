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


Standard training
===========

In ``Train``,  a default training script ``default_training.y`` should be kept up to date.
Run it with:
``python3 default_training.py <path to your converted dataCollection.djcdc file> <output dir>``
Be careful, a bunch of plots will be produced (this  can be changed in the lower part of the script, just increase the number of batches after which a plot is made.



Analysis and plots
===========
Assuming that the prediction files are in `/mnt/ceph/users/sqasim/Workspace/NextCal/HGCalML/srq/test_files/out`, use this to make plots


``python3 analyse_and_plot_clustering_in_hgcal_using_object_condensation.py /mnt/ceph/users/sqasim/Workspace/NextCal/HGCalML/srq/test_files/out -p jan_9_14_2.pdf -v 10 --analysisoutpath dump.bin``

This will also generate dump.bin file to dump plot data. You can use it directly to make plots faster using:

``python3 plot_from_dump.py dump.bin jan_9_14.pdf``
