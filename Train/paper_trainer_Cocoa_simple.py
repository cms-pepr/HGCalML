"""
Training script based on paper_trainer_noRSU_fixed_jk.py
"""

import sys
from argparse import ArgumentParser

# import wandb
from DeepJetCore.wandb_interface import wandb_wrapper as wandb
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout
from tensorflow.keras import Model


import training_base_hgcal
from Layers import LLRegulariseGravNetSpace, DummyLayer
from Layers import ProcessFeaturesCocoa
from Layers import RaggedGravNet
from Layers import PlotCoordinates
from Layers import TranslationInvariantMP
from Layers import LLExtendedObjectCondensation5
from Layers import LLExtendedObjectCondensation3
from model_blocks import condition_input
from model_blocks import extent_coords_if_needed
from model_blocks import create_outputs
from callbacks import plotClusterSummary
from tensorflow.keras.layers import BatchNormalization

####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('--run_name', help="wandb run name", default="test")
parser.add_argument('--no_wandb', help="Don't use wandb", action='store_true')
parser.add_argument('--wandb_project', help="wandb_project", default="Paper_Models")
#parser.add_argument('--lr', help="learning rate", default="0.0001")

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


PLOT_FREQUENCY = 600

DENSE_ACTIVATION = 'elu'
DENSE_INIT = "he_normal"
d_shape = 64
DENSE_REGULARIZER = tf.keras.regularizers.l2(1e-9)
NEIGHBOURS = [64,64,64,64]

###############################################################################
### Define Model ##############################################################
###############################################################################

def GravNet(name,
                       x, cprime, hit_energy, t_idx,
                       rs, 
                       d_shape, 
                       n_neighbours,
                       debug_outdir, 
                       plot_debug_every, 
                       space_reg_strength=-1e-2,
                       ):
    
    x= Dense(d_shape,activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
            kernel_regularizer=DENSE_REGULARIZER)(x)
    x= Dense(d_shape,activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
            kernel_regularizer=DENSE_REGULARIZER)(x)
    x= Dense(d_shape,activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
            kernel_regularizer=DENSE_REGULARIZER)(x)
    
    x = BatchNormalization()(x)
    
    x = Concatenate()([cprime, x])
    
    xgn, gncoords, gnnidx, gndist = RaggedGravNet(
                name = "GravNet_"+name, # 76929, 42625, 42625
            n_neighbours=n_neighbours,
            n_dimensions=4,
            n_filters=d_shape,
            n_propagate=d_shape,
            coord_initialiser_noise=1e-3,
            feature_activation='elu',
            )([x, rs])

    if space_reg_strength > 0:
        gndist = LLRegulariseGravNetSpace(name=f'gravnet_coords_reg_{name}' , 
                                          record_metrics=True,
                                          scale=space_reg_strength)([gndist, cprime, gnnidx])
    
    gncoords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir = debug_outdir,
            name=f'gncoords_{name}'
            )([gncoords, hit_energy, t_idx, rs])
    
    x = DummyLayer()([x, gncoords]) #just so the branch is not optimised away, anyway used further down
    x = BatchNormalization()(x)
    
    return Concatenate()([xgn, x])




def config_model(Inputs, td, debug_outdir=None, plot_debug_every=2000):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step ###################################################
    ###########################################################################

    orig_input = td.interpretAllModelInputs(Inputs)
    pre_processed = condition_input(orig_input, no_scaling=True, no_prime=False, new_prime=True)

    prime_coords = pre_processed['prime_coords']

    c_coords = prime_coords
    is_track = pre_processed['is_track']
    rs = pre_processed['row_splits']
    energy = pre_processed['rechit_energy']
    t_idx = pre_processed['t_idx']
    x = pre_processed['features']

    x = ProcessFeaturesCocoa()(x)

    c_coords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        name='input_c_coords',
        )([c_coords, energy, t_idx, rs])
    
    c_coords = extent_coords_if_needed(c_coords, x, 3)

    x = Concatenate()([x, c_coords, is_track])
    x = Dense(64, name='dense_pre_loop', activation=DENSE_ACTIVATION)(x)

    allfeat = []
    print("Available keys: ", pre_processed.keys())

    ###########################################################################
    ### Loop over GravNet Layers ##############################################
    ###########################################################################


    for i in range(len(NEIGHBOURS)):

        
        x = GravNet(f'gncomb_{i}', x, prime_coords, energy, t_idx, rs, 
                               d_shape, NEIGHBOURS[i], debug_outdir, plot_debug_every, space_reg_strength=-1e-2)

        #x = LayerNormalization()(x)
        allfeat.append(x)
        if len(allfeat) > 1:
            x = Concatenate()(allfeat)
        else:
            x = allfeat[0]


    ###########################################################################
    ### Create output of model and define loss ################################
    ###########################################################################


    x = Dense(128,
              name=f"dense_final_{1}",
              activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = BatchNormalization()(x)
    x = Dense(64,
              name=f"dense_final_{2}",
              activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = Dense(64,
              name=f"dense_final_{3}",
              activation=DENSE_ACTIVATION, kernel_initializer=DENSE_INIT,
              kernel_regularizer=DENSE_REGULARIZER)(x)  
    x = BatchNormalization()(x)  

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x,
                n_ccoords=3,
                fix_distance_scale=True,
                is_track=is_track,
                set_track_betas_to_one=True)


    pred_beta = LLExtendedObjectCondensation3(
            scale=1.,
            use_energy_weights=True,
            record_metrics=True,
            print_loss=train.args.no_wandb,
            name="ExtendedOCLoss",
            implementation = "hinge",
            beta_loss_scale = 1.0,
            too_much_beta_scale = 0.0,
            energy_loss_weight = 0.4,
            classification_loss_weight = 0.4,
            position_loss_weight =  0.0,
            timing_loss_weight = 0.0,
            q_min = 1.0,
            use_average_cc_pos = 0.9999)(
                    [pred_beta, pred_ccoords, pred_dist, pred_energy_corr, pred_energy_low_quantile,
                        pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id, energy,
                        pre_processed['t_idx'] , pre_processed['t_energy'] , pre_processed['t_pos'] ,
                        pre_processed['t_time'] , pre_processed['t_pid'] , pre_processed['t_spectator_weight'],
                        pre_processed['t_fully_contained'], pre_processed['t_rec_energy'],
                        pre_processed['t_is_unique'], pre_processed['row_splits']])

    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir = debug_outdir,
        name='condensation'
        )([pred_ccoords, pred_beta, pre_processed['t_idx'], rs])

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
        'row_splits': pre_processed['row_splits'],
        # 'no_noise_sel': pre_processed['no_noise_sel'],
        # 'no_noise_rs': pre_processed['no_noise_rs'],
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
    train.applyFunctionToAllModels(RaggedGravNet.set_all_gn_space_trainable, False) #start with fixed GN space
    train.setCustomOptimizer(tf.keras.optimizers.Adam())#clipnorm=1.))
    train.compileModel(learningrate=1e-4)
    train.keras_model.summary()
    train.saveModel(train.outputDir+'/model_init.h5')

###############################################################################
### Callbacks #################################################################
###############################################################################


samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH = ""
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]


cb = []#[NanSweeper()] #this takes a bit of time checking each batch but could be worth it

cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=1000
        )
    ]


# cb += [wandbCallback()]

###############################################################################
### Actual Training ###########################################################
###############################################################################

train.change_learning_rate(5e-3)
train.trainModel(
        nepochs=2,
        batchsize=50000,
        add_progbar=pre_args.no_wandb,
        additional_callbacks=cb,
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
train.compileModel(learningrate=2e-4)
print('entering second training phase')
train.trainModel(
        nepochs=5,
        batchsize=50000,
        add_progbar=pre_args.no_wandb,
        additional_callbacks=cb,
        collect_gradients = 4
        )


#recompile
train.compileModel(learningrate=1e-4)
print('entering third training phase')
train.trainModel(
        nepochs=10,
        batchsize=50000,
        add_progbar=args.no_wandb,
        additional_callbacks=cb,
        collect_gradients = 4
        )