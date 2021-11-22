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
from lossLayers import LLFullObjectCondensation, LLClusterCoordinates

from model_blocks import create_outputs
from GravNetLayersRagged import MultiBackScatter,LNC2,EdgeCreator, EdgeSelector, GroupScoreFromEdgeScores,NoiseFilter,LNC,ProcessFeatures,SoftPixelCNN, RaggedGravNet, DistanceWeightedMessagePassing
from Layers import CreateTruthSpectatorWeights, ManualCoordTransform,RaggedGlobalExchange,LocalDistanceScaling,CheckNaN,NeighbourApproxPCA,GraphClusterReshape, SortAndSelectNeighbours, LLLocalClusterCoordinates,DistanceWeightedMessagePassing,CollectNeighbourAverageAndMax,CreateGlobalIndices, LocalClustering, SelectFromIndices, MultiBackScatter, KNN, MessagePassing, RobustModel
from Layers import GausActivation,GooeyBatchNorm #make a new line
from model_blocks import create_outputs, noise_pre_filter
from Regularizers import AverageDistanceRegularizer



from debugLayers import PlotCoordinates

'''

make this about coordinate shifts


'''


td = TrainData_NanoML()


def gravnet_model(Inputs,
                  viscosity=0.8,
                  print_viscosity=False,
                  fluidity_decay=5e-4,  # reaches after about 7k batches
                  max_viscosity=0.95,
                  debug_outdir=None
                  ):
    # Input preprocessing below. Not much to change here

    feat, t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = td.interpretAllModelInputs(
        Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(feat)

    t_spectator_weight = CreateTruthSpectatorWeights(threshold=3.,
                                                     minimum=1e-1,
                                                     active=True
                                                     )([t_spectator, t_idx])
    orig_t_spectator_weight = t_spectator_weight

    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits

    feat_norm = ProcessFeatures()(feat)
    energy = SelectFeatures(0, 1, name="energy_selector")(feat)
    time = SelectFeatures(8, 9)(feat_norm)
    orig_coords = SelectFeatures(5, 8)(feat_norm)

    x = feat_norm
    sel_gidx = gidx_orig

    allfeat = [x]
    backgatheredids = []
    scatterids = []
    backgathered = []
    backgathered_coords = []
    energysums = []

    # here the actual network starts

    ############## Keep this part to reload the noise filter with pre-trained weights for other trainings
    
    # really simple real coordinates
    coords = orig_coords
    
    #write out noise filter explicitly
    x = Concatenate()([coords,x])
    x, coords, nidx, dist = RaggedGravNet(n_neighbours=64,
                                          n_dimensions=3,
                                          n_filters=64,
                                          n_propagate=32,
                                          return_self=True,
                                          name='initial_nf_noise_gn')([x, rs])
    
    #shift coordinates for the first time
    coordsdiff = WeightedNeighbourMeans()([coords,energy,dist,nidx])
    coords = Add()([coords,coordsdiff])
    
    noise_score = Dense(1, activation='sigmoid', name='initial_nf_noise_score')(x)
    other = [x, coords, energy, sel_gidx, t_spectator_weight, t_idx]
    rs, bg, *other = NoiseFilter(threshold=0.1,#
                                 loss_enabled=True, 
                                 loss_scale=1.,
                                 print_loss=True,
                                 return_backscatter=True,
                                 print_reduction=True,
                                 name='initial_nf_noise_filter' #so that we can load it back
                                 )([ noise_score, rs] +
                                        other +
                                        [t_idx ])
                                 
    full_noise_score = StopGradient()(noise_score)                             
    scatterids.append(bg)     
    #now selected only                             
    x, coords, energy, sel_gidx, t_spectator_weight, t_idx = other  
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)                                
              
                         
    #################### noise filter done. proceed to reduction iterations
    lnc_coords = coords
    
    total_iterations=2

    for i in range(total_iterations):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])
        
        
        K=64
        #create clustering coordinates
        lnc_coords = Dense(3, name='newcoords' + str(i),
                        kernel_initializer=EyeInitializer(stddev=0.01), use_bias=False,
                            )(Concatenate()([lnc_coords, x]))
        
        cnidx, cdist = KNN(K=K, radius=-1.0)([lnc_coords, rs])
        cdist = LLLocalClusterCoordinates(
            print_loss=True,
            scale=1.
            )([cdist, cnidx, t_idx, t_spectator_weight])
        
        #create distance weights
        x_preedge = Dense(4,activation='relu')(x) 
        preedges = EdgeCreator(addself=True)([cnidx, x_preedge])
        distmod = Dense(4,activation='relu')(preedges)
        distmod = Dense(1,activation='sigmoid')(distmod)
        distmod = ScalarMultiply(2.)(distmod)
        distmod = Reshape((K+1,))(distmod)
        cdist = Multiply()([cdist,distmod])
        
        #add loss to distances
        cdist = LLLocalClusterCoordinates(
            print_loss=True,
            scale=1.
            )([cdist, cnidx, t_idx, t_spectator_weight])
                
        #select closest
        cdist, cnidx = SortAndSelectNeighbours(K=16)([cdist, cnidx])
        #exchange information twice (beyond neighbourhood boundaries
        x = MessagePassing([32,32])([x, cnidx])
        
        hier = Dense(1)(Concatenate()([cdist,x]))
        
        ########## this is the reduction block
        lnc_input = [hier, cdist, cnidx, rs, t_spectator_weight, t_idx]
        lnc_flatten = []
        lnc_sum = [energy, hier]
        lnc_meanmax = []
        lnc_sel = [x,lnc_coords,sel_gidx,t_spectator_weight,t_idx]
        
        lnc_out, lnc_flat_out, lnc_sum_out, lnc_meanmax_out, lnc_sel_out = LNC2( 
                               threshold=0.5,# overmerging is not horrible
                               loss_scale=0.1,  
                               print_reduction=True,
                               loss_enabled=True,  # all gradients already exist
                               distance_loss_scale=.1, # alreade has a gradient
                               print_loss=True, 
                               print_output_shape=False, #useful for finding memory hogs
                               return_backscatter=True,
                               
                               name='LNC2_' + str(i))(
                                   [lnc_input, 
                                    lnc_flatten, 
                                    lnc_sum, 
                                    lnc_meanmax, 
                                    lnc_sel] )
                               
        nn, rs, bidxs = lnc_out
        energy, hier = lnc_sum_out
        x,lnc_coords,sel_gidx,t_spectator_weight,t_idx = lnc_sel_out
        
        scatterids.append(bidxs)
        
        ### reduction done
        
        #exchange information
        x_gn, coords, nidx, dist = RaggedGravNet(n_neighbours=32,
                                                 n_dimensions=3,
                                                 n_filters=128,
                                                 n_propagate=64)([x, rs])

        x_dmp = DistanceWeightedMessagePassing(4*[32])([x, nidx, dist])
        x_dmp = Dense(64, activation='relu')(x_dmp)
        x_mp = MessagePassing(4*[32])([x_dmp, nidx])
        x_mp = Dense(64, activation='relu')(x_mp)
        
        x = Concatenate()([x_gn,x_dmp,x_mp, energy,hier,coords,dist])
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        
        
        weight = Add()([energy,ScalarMultiply(0.01)(Dense(1)(x))])
        coordsdiff = WeightedNeighbourMeans()([coords,weight,dist, nidx])
        coordsdiff = Add()([coordsdiff, ScalarMultiply(0.01)(WeightFeatures()([coordsdiff,x]))])
        lnc_coords = Add()([coords,coordsdiff])
        
        x = Dense(64, activation='relu', name='dense_a_' + str(i))(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        x = Dense(64, activation='relu', name='dense_b_' + str(i))(x)
        x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
        x = Dense(64, activation='relu', name='dense_c_' + str(i))(x)
        
        
        
        allfeat.append(MultiBackScatter()([x, scatterids]))
        
        

    rs = row_splits  # important! here we are in non-reduced full graph mode again

    x = Concatenate(name='allconcat')([orig_coords]+[noise_score]+allfeat+energysums)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    #do one more exchange with all
    x, coords, nidx, dist = RaggedGravNet(n_neighbours=64,
                                                 n_dimensions=4,
                                                 n_filters=64,
                                                 n_propagate=64)([x, rs])
    x = MessagePassing(4*[32])([x, nidx])
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    
    ##now, here the later features will be undetermined for quite a while, so start with those
    x = Dense(64, activation='relu',
              name='alldense')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    x = Dense(64, activation='relu', kernel_initializer=EyeInitializer(stddev=0.01))(x)
    x = RaggedGlobalExchange()([x, row_splits])
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity, fluidity_decay=fluidity_decay)(x)
    x = Dense(48, activation='relu')(x)
    x = Concatenate()([lnc_coords,orig_coords,x, noise_score])  # we have it anyway
    #more normalisation here
    #x = GooeyBatchNorm(viscosity=viscosity/2., max_viscosity=max_viscosity, fluidity_decay=fluidity_decay/2.)(x)
    
    pred_beta, pred_ccoords, pred_dist, pred_energy, \
    pred_pos, pred_time, pred_id = create_outputs(x, feat, fix_distance_scale=False,
                                                  n_ccoords=3)

    # loss
    pred_beta = LLFullObjectCondensation(print_loss=True, scale=5.,
                                         energy_loss_weight=1e-3,
                                         position_loss_weight=1e-3,
                                         timing_loss_weight=1e-3,
                                         beta_loss_scale=1.,
                                         too_much_beta_scale=.01,
                                         use_energy_weights=True,
                                         q_min=2.5,
                                         div_repulsion=True,
                                         # cont_beta_loss=True,
                                         # beta_gradient_damping=0.999,
                                         # phase_transition=1,
                                         huber_energy_scale=3,
                                         use_average_cc_pos=0.5,  # smoothen it out a bit
                                         name="FullOCLoss"
                                         )(  # oc output and payload
        [pred_beta, pred_ccoords, pred_dist,
         pred_energy, pred_pos, pred_time, pred_id] +
        # truth information
        [orig_t_idx, orig_t_energy,
         orig_t_pos, orig_t_time, orig_t_pid,
         orig_t_spectator_weight,
         row_splits])

    model_outputs = [('pred_beta', pred_beta), 
                     ('pred_ccoords', pred_ccoords),
                     ('pred_energy', pred_energy),
                     ('pred_pos', pred_pos),
                     ('pred_time', pred_time),
                     ('pred_id', pred_id),
                     ('pred_dist', pred_dist),
                     ('row_splits', row_splits)]

    for i, (x, y) in enumerate(zip(backgatheredids, backgathered_coords)):
        model_outputs.append(('backgatheredids_' + str(i), x))
        model_outputs.append(('backgathered_coords_' + str(i), y))
    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)


import training_base_hgcal
train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)

if not train.modelSet():
    train.setModel(gravnet_model,debug_outdir=train.outputDir+'/intplots')
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

    train.compileModel(learningrate=1e-4,
                       loss=None)


verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


cb = []


cb += [plotClusteringDuringTraining(
    use_backgather_idx=8 + i,
    outputfile=train.outputDir + "/localclust/cluster_" + str(i) + '_',
    samplefile=samplepath,
    after_n_batches=500,
    on_epoch_end=False,
    publish=None,
    use_event=0)
    for i in [0, 2, 4]]

cb += [
    plotEventDuringTraining(
        outputfile=train.outputDir + "/condensation/c_"+str(i),
        samplefile=samplepath,
        after_n_batches=500,
        batchsize=200000,
        on_epoch_end=False,
        publish=None,
        use_event=i)
for i in range(5)
]

cb += [
    plotGravNetCoordsDuringTraining(
        outputfile=train.outputDir + "/localcoords/coord_" + str(i),
        samplefile=samplepath,
        after_n_batches=500,
        batchsize=200000,
        on_epoch_end=False,
        publish=None,
        use_event=0,
        use_prediction_idx=8+i,
    )
    for i in [1,3,5]  # between 16 and 21
]



#cb += build_callbacks(train)

#cb=[]
learningrate = 1e-3
nbatch = 30000

train.compileModel(learningrate=1e-3, #gets overwritten by CyclicLR callback anyway
                          loss=None,
                          metrics=None,
                          )

model, history = train.trainModel(nepochs=3,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/5.,
                                  max_lr = learningrate,
                                  step_size = 250)]+cb)

print("freeze BN")
# Note the submodel here its not just train.keras_model
for l in train.keras_model.model.layers:
    if 'gooey_batch_norm' in l.name:
        l.max_viscosity = 0.99
        l.fluidity_decay= 1e-4 #reaches constant 1 after about one epoch
    if 'FullOCLoss' in l.name:
        l.use_average_cc_pos = 0.1
        l.q_min = 2.
        l.cont_beta_loss=False
        l.energy_loss_weight=1e-2 #etc
        l.position_loss_weight=1e-2
    if 'edge_selector' in l.name:
        l.use_truth=False#IMPORTANT

#also stop GravNetLLLocalClusterLoss* from being evaluated
learningrate/=10.
nbatch = 120000

train.compileModel(learningrate=learningrate,
                          loss=None,
                          metrics=None)

model, history = train.trainModel(nepochs=121,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys), #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=verbosity,
                                  backup_after_batches=500,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/10.,
                                  max_lr = learningrate,
                                  step_size = 100)]+cb)
#

