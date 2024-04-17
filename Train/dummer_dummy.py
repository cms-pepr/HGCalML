"""
Flexible training script that should be mostly configured with a yaml config file
"""

import os
import pdb
import sys
import yaml
import shutil
from argparse import ArgumentParser

# import wandb
from DeepJetCore.wandb_interface import wandb_wrapper as wandb
from wandb_callback import wandbCallback
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout
from tensorflow.keras import Model

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from DeepJetCore.DJCLayers import StopGradient, ScalarMultiply

import training_base_hgcal
from Layers import ScaledGooeyBatchNorm2
from Layers import MixWhere
from Layers import RaggedGravNet
from Layers import PlotCoordinates
from Layers import DistanceWeightedMessagePassing, TranslationInvariantMP
from Layers import LLFillSpace
from Layers import LLExtendedObjectCondensation
from Layers import LLExtendedObjectCondensation2
from Layers import LLExtendedObjectCondensation3
from Layers import LLExtendedObjectCondensation4
from Layers import DictModel
from Layers import RaggedGlobalExchange
from Layers import SphereActivation
from Layers import Multi
from Layers import ShiftDistance
from LossLayers import LLDummy
# from Layers import LLRegulariseGravNetSpace
from Layers import SplitOffTracks, ConcatRaggedTensors
from Regularizers import AverageDistanceRegularizer
from Initializers import EyeInitializer
from model_blocks import tiny_pc_pool, condition_input
from model_blocks import extent_coords_if_needed
from model_blocks import create_outputs
from model_tools import apply_weights_from_path
from model_blocks import random_sampling_unit, random_sampling_block2
from noise_filter import noise_filter
from callbacks import plotClusteringDuringTraining
from callbacks import plotClusterSummary
from callbacks import NanSweeper, DebugPlotRunner


####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('configFile')
parser.add_argument('--run_name', help="wandb run name", default="test")
parser.add_argument('--no_wandb', help="Don't use wandb", action='store_true')
parser.add_argument('--wandb_project', help="wandb_project", default="Paper_Models")

train = training_base_hgcal.HGCalTraining(parser=parser)
CONFIGFILE = train.args.configFile

print(f"Using config File: \n{CONFIGFILE}")

with open(CONFIGFILE, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

N_CLUSTER_SPACE_COORDINATES = config['General']['n_cluster_space_coordinates']
N_GRAVNET_SPACE_COORDINATES = config['General']['n_gravnet_space_coordinates']
GRAVNET_ITERATIONS = len(config['General']['gravnet'])
LOSS_OPTIONS = config['LossOptions']
BATCHNORM_OPTIONS = config['BatchNormOptions']
DENSE_ACTIVATION = config['DenseOptions']['activation']
DENSE_REGULARIZER = tf.keras.regularizers.l2(config['DenseOptions']['kernel_regularizer_rate'])
DROPOUT = config['DenseOptions']['dropout']
DISTANCE_SCALE= bool(config['General']['fix_distance'])
loss_layer = LLExtendedObjectCondensation2

if "LossLayer" in config['General']:
    if config['General']['loss_layer'] == "2":
        loss_layer = LLExtendedObjectCondensation2
    elif config['General']['loss_layer'] == "3":
        loss_layer = LLExtendedObjectCondensation3
    elif config['General']['loss_layer'] == "4":
        loss_layer = LLExtendedObjectCondensation4
else:
    config['General']['loss_layer'] = 2


wandb_config = {
    "loss_implementation"           :   config['General']['oc_implementation'],
    "gravnet_iterations"            :   GRAVNET_ITERATIONS,
    "gravnet_space_coordinates"     :   N_GRAVNET_SPACE_COORDINATES,
    "cluster_space_coordinates"     :   N_CLUSTER_SPACE_COORDINATES,
    "loss_energy_weight"            :   config['LossOptions']['energy_loss_weight'],
    "loss_classification_weight"    :   config['LossOptions']['classification_loss_weight'],
    "loss_qmin"                     :   config['LossOptions']['q_min'],
    "loss_use_average_cc_pos"       :   config['LossOptions']['use_average_cc_pos'],
    "loss_too_much_beta_scale"      :   config['LossOptions']['too_much_beta_scale'],
    "loss_beta_scale"               :   config['LossOptions']['beta_loss_scale'],
    "batch_max_viscosity"           :   config['BatchNormOptions']['max_viscosity'],
    "dense_activation"              :   config['DenseOptions']['activation'],
    "dense_kernel_reg"              :   config['DenseOptions']['kernel_regularizer_rate'] ,
    "dense_dropout"                 :   config['DenseOptions']['dropout'],
    "distance_scale"                :   DISTANCE_SCALE,
    "LossLayer"                     :   config['General']['loss_layer']
}

for i in range(GRAVNET_ITERATIONS):
    wandb_config[f"gravnet_{i}_neighbours"] =config['General']['gravnet'][i]['n']
for i in range(len(config['Training'])):
    wandb_config[f"train_{i}_lr"] = config['Training'][i]['learning_rate']
    wandb_config[f"train_{i}_epochs"] = config['Training'][i]['epochs']
    wandb_config[f"train_{i}_batchsize"] = config['Training'][i]['batch_size']
    if i == 1:
        wandb_config[f"train_{i}+_max_visc"] = 0.999
        wandb_config[f"train_{i}+_fluidity_decay"] = 0.1

if not train.args.no_wandb:
    wandb.init(
        project=train.args.wandb_project,
        config=wandb_config,
    )
    wandb.save(sys.argv[0]) # Save python file
    wandb.save(train.args.configFile) # Save config file
else:
    wandb.active=False



###############################################################################
### Define Model ##############################################################
###############################################################################


def config_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
    """
    Function that defines the model to train
    """

    orig_input = td.interpretAllModelInputs(Inputs)
    pre_processed = condition_input(
            orig_input,
            no_scaling=True,
            no_prime=False)

    x = pre_processed['features']
    x = Dense(64, name='dense', activation='elu')(x)

    output = LLDummy(
            record_metrics=False,
            print_loss=False,
            name="Dummy",)(x)

    return Model(inputs=Inputs, outputs=output)


if not train.modelSet():
    train.setModel(
        config_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        )
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.))
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()

###############################################################################
### Callbacks #################################################################
###############################################################################

samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH = ""
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]
RECORD_FREQUENCY = 10
PLOT_FREQUENCY = 40


###############################################################################
### Actual Training ###########################################################
###############################################################################

shutil.copyfile(
    CONFIGFILE, os.path.join(sys.argv[3], "config.yaml")
    )

N_TRAINING_STAGES = len(config['Training'])
for i in range(N_TRAINING_STAGES):
    print(f"Starting training stage {i}")
    learning_rate = config['Training'][i]['learning_rate']
    epochs = config['Training'][i]['epochs']
    batch_size = config['Training'][i]['batch_size']
    train.change_learning_rate(learning_rate)
    print(f"Training for {epochs} epochs")
    print(f"Learning rate set to {learning_rate}")
    print(f"Batch size: {batch_size}")

    if i == 1:
        # change batchnorm
        for layer in train.keras_model.layers:
            if 'batchnorm' in layer.name:
                layer.max_viscosity = 0.999
                layer.fluidity_decay = 0.01
    train.trainModel(
        nepochs=epochs,
        batchsize=batch_size,
        add_progbar=True,
        additional_callbacks=[],
        )
