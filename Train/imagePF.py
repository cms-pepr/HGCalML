"""
Training script

python3 imagePF.py <input data collection.djcdc> 

"""

import os
import pdb
import sys
import yaml
import shutil
from argparse import ArgumentParser

# import wandb
from DeepJetCore.wandb_interface import wandb_wrapper as wandb
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Conv2D, Flatten, Reshape, BatchNormalization
from tensorflow.keras import Model
from LossLayers import LLExtendedObjectCondensation3

from training_base_hgcal import training_base_hgcal
from model_blocks import create_outputs
from Layers import SuperFlatten

from DebugLayers import PlotCoordinates

####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('--run_name', help="wandb run name", default="test")
parser.add_argument('--no_wandb', help="Don't use wandb", action='store_true')
parser.add_argument('--wandb_project', help="wandb_project", default="image_PF")

# get the args so far but ignore the other args to pass them on
# this prevents errors with wandb in layers

pre_args,_ = parser.parse_known_args() 

if not pre_args.no_wandb:
    wandb.init(
        project=pre_args.wandb_project,
        config={},
    )
    wandb.wandb().save(sys.argv[0]) # Save python file
else:
    wandb.active=False
#parses the rest of the arguments
train = training_base_hgcal.HGCalTraining(parser=parser)

PLOT_FREQUENCY = 8000 # a bit more than one 2 hours

N_CLUSTER_SPACE_COORDINATES=3

DENSE_ACTIVATION='elu'
DISTANCE_SCALE=False

loss_layer = LLExtendedObjectCondensation3

LOSS_OPTIONS = {
    "beta_loss_scale": 1.0,
    "too_much_beta_scale": 0.0,
    "energy_loss_weight": 1e-5,
    "classification_loss_weight": 0.,
    "position_loss_weight": 0.0,
    "timing_loss_weight": 0.0,
    "q_min": 1.0,
    "use_average_cc_pos": 0.9999,
}

###############################################################################
### Define Model ##############################################################
###############################################################################

def model(Inputs, td, debug_outdir=None, plot_debug_every=PLOT_FREQUENCY):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step ###################################################
    ###########################################################################

    orig_input = td.interpretAllModelInputs(Inputs)
    
    is_track = orig_input['is_track']
    energy = orig_input['rechit_energy']
    t_idx = orig_input['t_idx']
    x = orig_input['features']

    ###########################################################################
    ###########################################################################
    ###########################################################################

    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'
    x = Conv2D((5,5))(x)# padding = 'same'


  

    x, rs  = SuperFlatten(event_size=50*50)(x)#that's not really correct

    ###########################################################################
    ### Create output of model and define loss ################################
    ###########################################################################


    x = Dense(128,
              name=f"dense_final_{1}",
              activation=DENSE_ACTIVATION)(x)
    x = Dense(64,
              name=f"dense_final_{2}",
              activation=DENSE_ACTIVATION)(x)
    x = Dense(64,
              name=f"dense_final_{3}",
              activation=DENSE_ACTIVATION)(x)
    #x = BatchNormalization()(x)
    

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x,
                n_ccoords=N_CLUSTER_SPACE_COORDINATES,
                fix_distance_scale=not DISTANCE_SCALE,
                is_track=is_track,
                set_track_betas_to_one=True)


    pred_beta = loss_layer(
            scale=1.,
            use_energy_weights=True,
            record_metrics=True,
            print_loss=train.args.no_wandb,
            name="ExtendedOCLoss",
            implementation = 'hinge',
            **LOSS_OPTIONS)(
                    [pred_beta, pred_ccoords, pred_dist, pred_energy_corr, pred_energy_low_quantile,
                        pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id, energy,
                        orig_input['t_idx'] , orig_input['t_energy'] , orig_input['t_pos'] ,
                        orig_input['t_time'] , orig_input['t_pid'] , orig_input['t_spectator_weight'],
                        orig_input['t_fully_contained'], orig_input['t_rec_energy'],
                        orig_input['t_is_unique'], orig_input['row_splits']])

    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir = debug_outdir,
        name='condensation'
        )([pred_ccoords, pred_beta, orig_input['t_idx'], rs])

    model_outputs = {
        'pred_beta': pred_beta,
        'pred_ccoords': pred_ccoords,
        'pred_energy_corr_factor': pred_energy_corr,
        'pred_energy_low_quantile': pred_energy_low_quantile,
        'pred_energy_high_quantile': pred_energy_high_quantile,
        'pred_pos': pred_pos,
        'pred_time': pred_time,
        'pred_id': pred_id,
        'pred_dist': pred_dist,
        'rechit_energy': energy,
        'row_splits': rs,
        }

    # return DictModel(inputs=Inputs, outputs=model_outputs)
    return Model(inputs=Inputs, outputs=model_outputs)


###############################################################################
### Set up training ###########################################################
###############################################################################



if not train.modelSet():
    train.setModel(
        config_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        plot_debug_every=PLOT_FREQUENCY,
        )
    train.setCustomOptimizer(tf.keras.optimizers.Adam())#clipnorm=1.))
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()

###############################################################################
### Callbacks #################################################################
###############################################################################


samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH = ""
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]


cb = []#[NanSweeper()] #this takes a bit of time checking each batch but could be worth it


# cb += [wandbCallback()]

###############################################################################
### Actual Training ###########################################################
###############################################################################

batchsize = 300

train.change_learning_rate(1e-3)
train.trainModel(
        nepochs=10,
        batchsize=batchsize,
        add_progbar=pre_args.no_wandb,
        additional_callbacks=[],
        collect_gradients = 4 #average out more gradients
        )

train.change_learning_rate(2e-4)
train.trainModel(
        nepochs=1+2+10,
        batchsize=batchsize,
        add_progbar=pre_args.no_wandb,
        additional_callbacks=[],
        collect_gradients = 4 #average out more gradients
        )

# loop through model layers and turn  batch normalisation to fixed
def fix_batchnorm(m):
    for layer in m.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

#apply to all models
train.applyFunctionToAllModels(fix_batchnorm)
#recompile
train.compileModel(learningrate=1e-3)

train.change_learning_rate(3e-5)
train.trainModel(
        nepochs=1+2+20+10,
        batchsize=batchsize,
        add_progbar=False,
        additional_callbacks=[],
        collect_gradients = 4
        )