# Config Trainer

---

## Usage

The `config_trainer.py` script aims to allow testing of different architectures and training strategies and keeping track of their performance using the `mlflow` package. 

In general the training script is used as all previous training scripts. Currently the path to a configuration file in yaml format has to be hard-coded in the script as I did not want to mess with the argument parser form `training_base_hgcal`. 
The configuration file then describes the model in several sections. Typically the naming should identify what parameters are doing.

* General
	* type of preprocessing model to be used (currently only tested with noise-filter and pre-pooling)
	* path to that model
	* Use fill-space loss -> This at some point lead to errors, I would have to double check it
	* use-layer-normalization -> Not implemented yet, coming soon 

* Architecture
	* gravnet: Settings for every gravnet iteration (big loop). Currently the only settings we are potentially changing between gravnet iterations is the number of layers, but this can be extended.
	* message-passing: Filters to be used in the message passing loop. This will be identical for every gravnet iteration
	* dense*: Configuration for loops of dense layers at different points in the network. 

* Dense Options: Options to be used for a dense layer throughout the network. 

* Batchnorm Options: Options for the `ScaledGooeyBatchnorm2` if they are different from the default values

* Loss Options: Exactly what you would expect

* Training: Training is also done within a big loop. Anything that should be changed between sets of epochs can go here

* Plotting options


## MLFLOW

Trainings can be compared using `mlflow server --backend-store-uri mlruns` where `mlruns` is the directory created by the mlflow package. 
This directory will by default be created in the directory where the training script is started. 
One should be able to call this on a laptop while mounting `mlruns` remotely with `sshfs`. 
So far it sometimes seemed to be quite slow but it should work. 
It now also should register all options in the configuration dictionary as trackable parameters.  

