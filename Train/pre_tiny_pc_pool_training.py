'''


Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''

import globals
if False: #for testing
    globals.acc_ops_use_tf_gradients = True 
    globals.knn_ops_use_tf_gradients = True

import tensorflow as tf
# from K import Layer

#from datastructures import TrainData_NanoML

#from tensorflow.keras import Model
from Layers import DictModel, PlotCoordinates, Where
from tensorflow.keras.layers import Concatenate, Dense

from model_blocks import  tiny_pc_pool, condition_input
from GraphCondensationLayers import add_attention, PushUp
from callbacks import NanSweeper


plot_frequency= 40  # 150 #150 # 1000 #every 20 minutes approx
record_frequency = 20

reduction_target = 0.05
lr_factor = .2
nbatch = 170000 

no_publish = False

train_second = False
if train_second:
    lr_factor *= reduction_target
    nbatch = 170000
    
train_all = False

def pretrain_model(Inputs,
                   td, 
                   debugplots_after=record_frequency*plot_frequency, #10 minutes: ~600
                   debug_outdir=None,
                   publishpath=None):

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
    presel = condition_input(orig_inputs, no_scaling=True)
 
 
    
    presel['prime_coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name='pc_pool_coords_pre',
                                        publish=publishpath)(
                                            [presel['prime_coords'],
                                             presel['rechit_energy'], 
                                             presel['t_idx'],presel['row_splits']])
 
    trans,presel = tiny_pc_pool(presel,
                                reduction_target = reduction_target,
                          trainable=not train_second or train_all,
                          record_metrics = True,
                          publish=publishpath,
                          debugplots_after=debugplots_after,
                          debug_outdir=debug_outdir,
                          )
    
    
    presel['cond_coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name='pc_pool_cond_coords0',
                                        publish=publishpath)(
                                            [presel['cond_coords'],
                                             presel['rechit_energy'],#Where(0.5)([presel['is_track'],presel['rechit_energy']]), 
                                             presel['t_idx'],presel['row_splits']])
                                        
    
    presel['prime_coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name='pc_pool_post_prime0',
                                        publish=publishpath)(
                                            [presel['prime_coords'],
                                             presel['rechit_energy'],#Where(0.5)([presel['is_track'],presel['rechit_energy']]), 
                                             presel['t_idx'],presel['row_splits']])
    
    presel['select_prime_coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name='pc_pool_post_sel_prime',
                                        publish=publishpath)(
                                            [presel['select_prime_coords'],
                                             presel['rechit_energy'],#Where(0.5)([presel['is_track'],presel['rechit_energy']]), 
                                             presel['t_idx'],presel['row_splits']])
                                        
                                        
                                        
    if train_second:
        
        trans,presel = tiny_pc_pool(presel,
                                    #coords = coords,
                                    name='pre_graph_pool1',
                                    is_second = True,
                                    reduction_target=0.1,
                              trainable=True,
                              #coords = coords,
                              #low_energy_cut_target = 1.0,
                              record_metrics = True,
                              publish=publishpath,
                              debugplots_after=debugplots_after,
                              debug_outdir=debug_outdir
                              )
        
        
        presel['prime_coords'] = PlotCoordinates(plot_every=debugplots_after,
                                            outdir=debug_outdir,name='pc_pool_post_prime1',
                                            publish=publishpath)(
                                                [presel['prime_coords'],
                                                 presel['rechit_energy'],#Where(1.)([presel['is_track'],presel['rechit_energy']]), 
                                                 presel['t_idx'],presel['row_splits']])
                                            
        
        presel['cond_coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name='pc_pool_cond_coords1',
                                        publish=publishpath)(
                                            [presel['cond_coords'],
                                             presel['rechit_energy'],#Where(0.5)([presel['is_track'],presel['rechit_energy']]), 
                                             presel['t_idx'],presel['row_splits']])
    
    presel.update(trans) #put them all in
    #presel.pop('row_splits')
    return DictModel(inputs=Inputs, outputs=presel)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/June2023/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 

if no_publish:
    publishpath = None

print('will attempt to publish to',publishpath)

if not train.modelSet():
    train.setModel(pretrain_model,
                   td = train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots',
                   publishpath=publishpath)
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam(clipnorm=1.))
    #
    train.compileModel(learningrate=1e-4)
    
    train.keras_model.summary()
    
    #start somewhere
    #from model_tools import apply_weights_from_path
    #import os
    #path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
    #train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    



verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
#publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/June2023/"+os.path.basename(os.path.normpath(train.outputDir))

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback



cb = [
    
    NanSweeper(),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*_reduction'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/hit_reduction_metrics.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*hits*','*lost*','*tracks*'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*_loss','*_accuracy'],
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/batchnorm.html',
        call_on_epoch=True,
        select_metrics='*norm*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    ]

#cb=[]

train.change_learning_rate(lr_factor*1e-2)
train.trainModel(nepochs=2, batchsize=nbatch,additional_callbacks=cb)

train.change_learning_rate(lr_factor*1e-3)
train.trainModel(nepochs=10, batchsize=nbatch,additional_callbacks=cb)

train.change_learning_rate(lr_factor*1e-4)
train.trainModel(nepochs=60, batchsize=nbatch,additional_callbacks=cb)

train.change_learning_rate(lr_factor*1e-5)
train.trainModel(nepochs=80, batchsize=nbatch,additional_callbacks=cb)

exit() #done
#nbatch = 150000 
train.change_learning_rate(3e-4)
train.trainModel(nepochs=10,batchsize=nbatch,additional_callbacks=cb)

print('reducing learning rate to 1e-4')
train.change_learning_rate(1e-5)
#nbatch = 200000 

train.trainModel(nepochs=100,batchsize=nbatch,additional_callbacks=cb)
