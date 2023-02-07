'''


Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''

import tensorflow as tf
# from K import Layer

from datastructures import TrainData_NanoML

#from tensorflow.keras import Model
from Layers import DictModel

from model_blocks import  pre_condensation_model, mini_pre_condensation_model

K=12 #12

plot_frequency= 20  # 150 #150 # 1000 #every 20 minutes approx
record_frequency = 3

def pretrain_model(Inputs,
                   td, 
                   debugplots_after=record_frequency*plot_frequency, #10 minutes: ~600
                   debug_outdir=None,
                   publishpath=None):

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
 
    presel = mini_pre_condensation_model(orig_inputs,
                           record_metrics=True,
                           trainable=True,
                           t_d=0.5, # just starting point
                           t_b=0.6, # just starting point
                           q_min=1.,
                           purity_target=0.96,
                           condensation_mode = 'std', # std, precond, pushpull, simpleknn
                           noise_threshold=0.15,
                           print_batch_time=False,
                           condensate=True,
                           cluster_dims = 3,
                           cleaning_threshold=0.5,
                           debug_outdir=debug_outdir,
                           debugplots_after=debugplots_after,
                           publishpath=publishpath
                           )
    presel.pop('noise_backscatter')
    return DictModel(inputs=Inputs, outputs=presel)

import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/Sept2022/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 

print('will attempt to publish to',publishpath)

if not train.modelSet():
    train.setModel(pretrain_model,
                   td = train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots',
                   publishpath=publishpath)
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
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
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback



cb = [
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency= record_frequency,
        plot_frequency = plot_frequency,
        select_metrics=['*_reduction', '*_purity','*_cleaned_fraction','*contamination'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    #simpleMetricsCallback(
    #    output_file=train.outputDir+'/hit_reduction_metrics.html',
    #    record_frequency= record_frequency,
    #    plot_frequency = plot_frequency,
    #    select_metrics=['*reduction*hits*','*_reduction*lost*'],#includes time
    #    publish=publishpath #no additional directory here (scp cannot create one)
    #    ),
    #
    simpleMetricsCallback(
        output_file=train.outputDir+'/noise_metrics.html',
        record_frequency= record_frequency,
        plot_frequency = plot_frequency,
        select_metrics=['*noise*accuracy','*noise*loss','*noise*reduction','*purity','*efficiency'],
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/time.html',
        record_frequency= 2.*record_frequency,#doesn't change anyway
        plot_frequency = plot_frequency,
        select_metrics='*time*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency= record_frequency,
        plot_frequency = plot_frequency,
        select_metrics=['*_loss','*simple_knn_oc*'],
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    #simpleMetricsCallback(
    #    output_file=train.outputDir+'/gooey.html',
    #    record_frequency= record_frequency,
    #    plot_frequency = plot_frequency,
    #    select_metrics='*gooey*',
    #    publish=publishpath #no additional directory here (scp cannot create one)
    #    ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/oc_thresh.html',
        record_frequency= record_frequency,
        plot_frequency = plot_frequency,
        select_metrics='*_ll_*oc_thresholds*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    ]

#cb=[]
nbatch = 150000 
train.change_learning_rate(5e-4)
train.trainModel(nepochs=1, batchsize=nbatch,additional_callbacks=cb)

nbatch = 150000 
train.change_learning_rate(3e-5)
train.trainModel(nepochs=10,batchsize=nbatch,additional_callbacks=cb)

print('reducing learning rate to 1e-4')
train.change_learning_rate(1e-5)
nbatch = 200000 

train.trainModel(nepochs=100,batchsize=nbatch,additional_callbacks=cb)
