'''

Compatible with the dataset here:
/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/Oct2021_production/Gun20Part_CHEPDef_NoPropagate/NanoML
and (soon)
/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/Oct2021_production/Gun20Part_CHEPDef_NoPropagate/NanoMLTracks

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/OctProd/NanoML

not compatible with datasets before end of October 2021

'''
from callback_wrappers import build_callbacks
from experiment_database_manager import ExperimentDatabaseManager
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np
from tensorflow.keras.layers import Reshape,BatchNormalization, Dropout, Add
from LayersRagged  import RaggedConstructTensor
from GravNetLayersRagged import WeightFeatures,WeightedNeighbourMeans,DownSample, CreateIndexFromMajority, ProcessFeatures, SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from initializers import EyeInitializer
from tensorflow.keras.layers import Multiply, Dense, Concatenate, GaussianDropout
from datastructures import TrainData_NanoML

from plotting_callbacks import plotEventDuringTraining, plotGravNetCoordsDuringTraining, plotClusteringDuringTraining, plotClusterSummary
from DeepJetCore.DJCLayers import StopGradient,ScalarMultiply, SelectFeatures, ReduceSumEntirely

from clr_callback import CyclicLR

from model_blocks import create_outputs
from GravNetLayersRagged import MultiBackScatter,EdgeCreator, EdgeSelector
from GravNetLayersRagged import GroupScoreFromEdgeScores,NoiseFilter
from GravNetLayersRagged import ProcessFeatures,SoftPixelCNN, RaggedGravNet
from GravNetLayersRagged import DistanceWeightedMessagePassing,MultiBackScatterOrGather

from GravNetLayersRagged import NeighbourGroups,AccumulateNeighbours,SelectFromIndices
from GravNetLayersRagged import RecalcDistances, ElementScaling, RemoveSelfRef, CastRowSplits

from Layers import CreateTruthSpectatorWeights, ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CreateGlobalIndices, SelectFromIndices, MultiBackScatter, KNN, MessagePassing
from Layers import GausActivation,GooeyBatchNorm #make a new line
from model_blocks import create_outputs, noise_pre_filter
from Regularizers import AverageDistanceRegularizer

from model_blocks import first_coordinate_adjustment, reduce, pre_selection_model_full
from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits, pre_selection_staged

from lossLayers import LLNeighbourhoodClassifier, LLNotNoiseClassifier
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates,LLEdgeClassifier
from lossLayers import LLFillSpace

from debugLayers import PlotCoordinates

from Layers import DictModel

'''

make this about coordinate shifts


'''




def gravnet_model(Inputs,
                  td,
                  viscosity=0.1,
                  print_viscosity=False,
                  fluidity_decay=1e-1,  # reaches after about 7k batches
                  max_viscosity=0.995,
                  debug_outdir=None,
                  plot_debug_every=1000,
                  ):
    # Input preprocessing below. Not much to change here

    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
    
    
    orig_t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([orig_inputs['t_spectator'], 
                                                        orig_inputs['t_idx']])
                                                     
    orig_inputs['t_spectator_weight'] = orig_t_spectator_weight                                                 
    #can be loaded - or use pre-selected dataset (to be made)
    pre_selection = pre_selection_model_full(orig_inputs)
    
    #just for info what's available
    print([k for k in pre_selection.keys()])
                                          
    ########## from here on everything is based on the pre-selection; only extend at the very end for the loss
    
    t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([pre_selection['t_spectator'], 
                                                        pre_selection['t_idx']])
    rs = pre_selection['rs']
                               
    x_in = Concatenate()([pre_selection['coords'],
                          pre_selection['features'],
                          pre_selection['addfeat']])
                           
    x = x_in
    energy = pre_selection['energy']
    coords = pre_selection['coords']
    t_idx = pre_selection['t_idx']
    
    
    ##################### now the actual model goes below
    
        
    allfeat = []
    
    n_cluster_space_coordinates = 3
    
    #extend coordinates already here if needed
    coords = extent_coords_if_needed(coords, x, n_cluster_space_coordinates)
    
    pre_coords = coords
    total_iterations=2
    

    for i in range(total_iterations):
            
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation='elu')(x)
        x = GooeyBatchNorm(viscosity=viscosity, 
                           name='gooey_pre_pre_gn_'+str(i),
                           max_viscosity=max_viscosity, 
                           fluidity_decay=fluidity_decay,
                           record_metrics=True)(x)
        x = Dense(64,activation='elu')(x)
        x = Dense(64,activation='elu')(x)
        x = GooeyBatchNorm(viscosity=viscosity, 
                           name='gooey_pre_gn_'+str(i),
                           max_viscosity=max_viscosity, 
                           fluidity_decay=fluidity_decay,
                           record_metrics=True)(x)
        ### reduction done
        
        n_dimensions = 6
        #in standard configuration with i<2
        n_neighbours = 128
        
        #exchange information, create coordinates
        x = Concatenate()([coords,coords,x])
        x, gncoords, gnnidx, gndist = RaggedGravNet(n_neighbours=n_neighbours,
                                                 n_dimensions=n_dimensions,
                                                 n_filters=128,
                                                 n_propagate=64,
                                                 coord_initialiser_noise=1e-3,
                                                 feature_activation=None,
                                                 record_metrics=True,
                                                 use_approximate_knn=True #faster on reduced data for now
                                                 )([x, rs])
                                                 
        #just keep them in a reasonable range  
        #safeguard against diappearing gradients on coordinates                                       
        gndist = AverageDistanceRegularizer(strength=1e-2,
                                            record_metrics=True,
                                            )(gndist)
        
        
        
        gncoords = PlotCoordinates(plot_debug_every, outdir = debug_outdir)([gncoords, 
                                                                    energy,
                                                                    t_idx,
                                                                    rs])
        x = Concatenate()([StopGradient()(ScalarMultiply(1e-6)(gncoords)),x])

        if True:
            ld=gndist
            for j in range(4): #consider different ranges
                dscaling = Dense(1)(x)
                ld = LocalDistanceScaling(record_metrics=True)([ld,dscaling])
                ld = AverageDistanceRegularizer(strength=1e-3,
                                            record_metrics=True,
                                            name='average_distance_scaled_'+str(i)+'_'+str(j)
                                            )(ld)
                x = DistanceWeightedMessagePassing([32,16])([x,gnnidx,ld])
            
        else: #in standard configuration with i<2
            x = DistanceWeightedMessagePassing([64,64,32,32,16,16,8,8])([x,gnnidx,gndist])
            
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, 
                           fluidity_decay=fluidity_decay,
                           name='gooey_post_mp_gn_'+str(i),
                           record_metrics=True)(x)
        
        #                     
        #compress output
        x = Dense(96,activation='elu')(x)
        
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, 
                           fluidity_decay=fluidity_decay,
                           name='gooey_post_pre_pre_merge_'+str(i),
                           record_metrics=True)(x)
                           
        x = Dense(64,activation='elu')(x)
        x = Dense(64,activation='elu')(x)
        
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, 
                           fluidity_decay=fluidity_decay,
                           name='gooey_post_pre_merge_'+str(i),
                           record_metrics=True)(x)
        
        #allfeat.append(MultiBackScatterOrGather()([x, scatterids]))
        allfeat.append(x)
        
    
    ## add coordinate differences
    #coords = Add()([coords, ScalarMultiply(-1.)(pre_coords)])
    
    
    ####### back to non-reduced space
    
    #extend coordinate list
    #coords = MultiBackScatterOrGather()([coords, scatterids])
    #pre_coords = extent_coords_if_needed(pre_selection['orig_dim_coords'], x, n_cluster_space_coordinates)
    #coords = Add()([coords,pre_coords])
    
    x = Concatenate()(allfeat+[pre_selection['not_noise_score']])
    #do one more exchange with all
    x = Dense(64,activation='elu')(x)
    x = Dense(64,activation='elu')(x)
    x = Dense(64,activation='elu')(x)
    x = GooeyBatchNorm(viscosity=viscosity, 
                       max_viscosity=max_viscosity, 
                       fluidity_decay=fluidity_decay,
                       name='gooey_pre_out')(x)
    x = Concatenate()([coords]+[x])
    
    pred_beta, pred_ccoords, pred_dist, pred_energy_corr, \
    pred_pos, pred_time, pred_id = create_outputs(x, pre_selection['unproc_features'], 
                                                  fix_distance_scale=False,
                                                  scale_energy=False,
                                                  energy_factor=True,
                                                  wide_distance_scale=True,
                                                  n_ccoords=n_cluster_space_coordinates)
    
    # loss
    pred_beta = LLFullObjectCondensation(print_loss=False, 
                                         record_metrics=True,
                                         scale=4.,
                                         energy_loss_weight=1.,
                                         position_loss_weight=1e-6,
                                         timing_loss_weight=1e-2,
                                         classification_loss_weight=1e-5,
                                         beta_loss_scale=3.,
                                         too_much_beta_scale=.0001,
                                         use_energy_weights=True,
                                         alt_energy_weight=True,
                                         q_min=0.5,
                                         #noise_q_min=2.0,
                                         #div_repulsion=True,
                                         # cont_beta_loss=True,
                                         # beta_gradient_damping=0.999,
                                         # phase_transition=1,
                                         #huber_energy_scale=0.1,
                                         use_average_cc_pos=0.3,  # smoothen it out a bit
                                         name="FullOCLoss"
                                         )(  # oc output and payload
        [pred_beta, pred_ccoords, pred_dist,
         pred_energy_corr, pred_pos, pred_time, pred_id] +
        [energy]+
        # truth information
        [pre_selection['t_idx'] ,
         pre_selection['t_energy'] ,
         pre_selection['t_pos'] ,
         pre_selection['t_time'] ,
         pre_selection['t_pid'] ,
         pre_selection['t_spectator_weight'] ,
         pre_selection['rs']])
                                         
    #fast feedback
    pred_ccoords = PlotCoordinates(plot_debug_every, outdir = debug_outdir,
                    name='condensation')([pred_ccoords, pred_beta,pre_selection['t_idx'],
                                          rs])                                    

    #model_outputs = [('pred_beta', pred_beta), 
    #                 ('pred_ccoords', pred_ccoords),
    #                 ('pred_energy_corr_factor', pred_energy_corr),
    #                 ('pred_pos', pred_pos),
    #                 ('pred_time', pred_time),
    #                 ('pred_id', pred_id),
    #                 ('pred_dist', pred_dist),
    #                 ('row_splits', row_splits)]
    #
    
    model_outputs = re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist,
        dict_output=True
        )
    return DictModel(inputs=Inputs, outputs=model_outputs)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(redirect_stdout=False)

if not train.modelSet():
    train.setModel(gravnet_model,
                   td=train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    train.setCustomOptimizer(tf.keras.optimizers.Adam(
        #larger->slower forgetting
        #beta_1: linear
        #beta_2: sq
        #make it slower for our weird fluctuating batches
        #beta_1=0.99, #0.9
        #beta_2=0.99999 #0.999
        #clipnorm=0.001
        #amsgrad=True,
        #epsilon=1e-2
        ))

    #get pretrained preselection weights
    
    from model_tools import apply_weights_from_path
    import os
    path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_multigrav/KERAS_model.h5'
    train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    
    #
    train.compileModel(learningrate=1e-4,
                       loss=None)


verbosity = 2
import os


samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/Dec2021/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 

cb = []



cb += [
    plotEventDuringTraining(
        outputfile=train.outputDir + "/condensation/c_"+str(i),
        samplefile=samplepath,
        after_n_batches=500,
        batchsize=200000,
        on_epoch_end=False,
        publish=publishpath+'_condensation_'+str(i),
        use_event=i)
for i in range(5)
]


from plotting_callbacks import plotClusterSummary
cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=samplepath,
        after_n_batches=800
        )
    ]


#cb += build_callbacks(train)

#cb=[]

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback

cb += [
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= 10,
        plot_frequency = 50,
        select_metrics='FullOCLoss_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/gooey_metrics.html',
        record_frequency= 10,
        plot_frequency = 70,
        select_metrics='gooey_*',
        publish=publishpath,
        smoothen=0
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/latent_space_metrics.html',
        record_frequency= 50,
        plot_frequency = 20,
        select_metrics='average_distance_*',
        publish=publishpath
        ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/slicing_knn_metrics.html',
        record_frequency= 10,
        plot_frequency = 100,
        publish=publishpath,
        select_metrics='*_bins'
    ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency= 10,
        plot_frequency = 50,
        publish=publishpath,
        select_metrics='*_reduction*'
    ),
    simpleMetricsCallback(
        output_file=train.outputDir+'/dist_scale.html',
        record_frequency= 50,
        plot_frequency = 10,
        publish=publishpath,
        select_metrics='*_dist_scale'
    )
    
    
    
    #simpleMetricsCallback(
    #    output_file=train.outputDir+'/global_metrics.html',
    #    record_frequency= 1,
    #    plot_frequency = 200,
    #    select_metrics='process*'
    #    ),
    ]
#cb=[]
learningrate = 5e-5
nbatch = 200000

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=11,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)


print(train.keras_model.summary())
print("ALMOST freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.layers:
    if 'gooey_batch_norm' in l.name:
        l.max_viscosity = 0.999999
        #l.fluidity_decay= 1e-4 #reaches constant 1 after about one epoch

#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=5.

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=121,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)


