'''
This is supposed to be a very basic model
'''
import pdb
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
import uuid

from datastructures import TrainData_NanoML
from Layers import CreateGlobalIndices, CreateTruthSpectatorWeights, MultiBackGather
from Layers import GooeyBatchNorm
from LayersRagged  import RaggedConstructTensor

from GravNetLayersRagged import ProcessFeatures
from DeepJetCore.DJCLayers import SelectFeatures, StopGradient

from model_blocks import noise_pre_filter

#### Old imports
from DeepJetCore.DJCLayers import StopGradient, SelectFeatures 
import matching_and_analysis
import training_base_hgcal

from experiment_database_manager import ExperimentDatabaseManager
from experiment_database_reading_manager import ExperimentDatabaseReadingManager

from GravNetLayersRagged import EdgeCreator, EdgeSelector, GroupScoreFromEdgeScores, LNC
from GravNetLayersRagged import RaggedGravNet, DistanceWeightedMessagePassing

from hyperparam_optimizer import OCHyperParamOptimizer
from tensorboard_manager import TensorBoardManager

from clr_callback import CyclicLR
from lossLayers import LLFullObjectCondensation 
from model_blocks import create_outputs, noise_pre_filter
from Layers import RaggedGlobalExchange, SortAndSelectNeighbours
from Layers import DistanceWeightedMessagePassing, MultiBackGather, KNN
from Layers import RobustModel, GooeyBatchNorm, ApproxPCA
from initializers import EyeInitializer

td=TrainData_NanoML()

def gravnet_model(Inputs,
                  viscosity=0.1,
                  print_viscosity=False,
                  fluidity_decay=1e-3, #reaches after about 7k batches
                  max_viscosity=0.95 
                  ):


    feat,  t_idx, t_energy, t_pos, t_time, t_pid, t_spectator, t_fully_contained, row_splits = td.interpretAllModelInputs(Inputs)
    orig_t_idx, orig_t_energy, orig_t_pos, orig_t_time, orig_t_pid, orig_row_splits = t_idx, t_energy, t_pos, t_time, t_pid, row_splits
    gidx_orig = CreateGlobalIndices()(feat)
    
    t_spectator_weight = CreateTruthSpectatorWeights(threshold = 1.21, 
                                                minimum=1e-2,
                                                active = True
                                                )([t_spectator,t_idx])
    orig_t_spectator_weight = t_spectator_weight

    _, row_splits = RaggedConstructTensor()([feat, row_splits])
    rs = row_splits

    feat_norm = ProcessFeatures()(feat)
    energy = SelectFeatures(0,1,name="energy_selector")(feat) # NOT normalized!
    time = SelectFeatures(8,9)(feat_norm)
    orig_coords = SelectFeatures(5,8)(feat_norm)

    x = feat_norm
    sel_gidx = gidx_orig

    allfeat=[x]
    backgatheredids=[]
    gatherids=[]
    backgathered = []
    backgathered_coords = []
    energysums = []

    #here the actual network starts

    ###>>> Filter noise
    coords = orig_coords
    other = [x, coords, energy, sel_gidx, t_spectator_weight, t_idx]
    coords, nidx, dist, noise_score, rs, bg, other = noise_pre_filter(
        x, coords, rs, other,  t_idx, threshold=0.025)
    x, coords, energy, sel_gidx, t_spectator_weight, t_idx = other
    
    noise_score = StopGradient()(noise_score)#just a pass through to the end
    
    gatherids.append(bg)
    backgathered_coords.append(MultiBackGather()([coords, gatherids]))
    ###>>> Noise filter part done
    
    # GooeyBatchNorm is expensive?
    x = Dense(64, activation='elu',name='pre_dense_a')(x)
    x = GooeyBatchNorm(viscosity=viscosity, 
                       max_viscosity=max_viscosity, 
                       fluidity_decay=fluidity_decay)(x)

    ccoords = coords


    #derive new coordinates for clustering
    ccoords = Dense(3,name='newcoords',
                    kernel_initializer=EyeInitializer(stddev=0.01),
                    use_bias=False
                    )(Concatenate()([ccoords,x]))
        
    nidx, cdist = KNN(K=10,radius=-1.0)([ccoords,rs])
        
    #here we use more neighbours to improve learning of the cluster space
    #this can be adjusted in the final trained model to be equal to 'cluster_neighbours'
    
    #create edge selection
    x_e = Dense(16,activation='elu')(x)
    edges = EdgeCreator()([nidx,x_e])
    edges = Dense(16,activation='elu')(edges)
    edges = Dense(8,activation='elu')(edges)
    edge_score = Dense(1,activation='sigmoid')(edges)
    nidx = EdgeSelector(threshold=0.6, 
                        loss_scale=.2, 
                        loss_enabled=True, 
                        print_loss=True)(
                            [nidx, edge_score]+
                            [t_spectator_weight, t_idx])
    
    
    hier = GroupScoreFromEdgeScores()([edge_score, nidx])
    
    x = Dense(128, activation='elu',name='dense_precl_a')(x)
    
    x_c, rs, bidxs,\
    sel_gidx, energy, x, t_idx, coords,ccoords,hier,t_spectator_weight = LNC(
                threshold = 0.001,#low because selection already done by edges
                loss_scale = .1,#more emphasis on the final OC loss
                print_reduction=True,
                loss_enabled=True, #loss still needed because of coordinate space
                use_spectators=True,
                sum_other=[1,6], #explicitly sum the energy and hier score
                distance_loss_scale=1.,
                print_loss=True,
                name='LNC'
                )( #this is needed by the layer
                [x, hier, ccoords, nidx, rs]+
                #these ones are selected accoring to the layer selection
                [sel_gidx, energy, x, t_idx, coords, ccoords, hier, t_spectator_weight]+
                #truth information passed to the layer to build loss
                [t_spectator_weight,t_idx])
    
    gatherids.append(bidxs)
    hier = StopGradient()(hier)
    x = Concatenate()([x,x_c,hier])
    
    #END OF IF CLAUSE
    
    x = Dense(128, activation='elu',name='dense_clc_a')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
    
    n_dimensions = 3 #make it plottable
    nneigh = 64 #this will be almost fully connected for last clustering step
    nfilt = 64
    nprop = 64

    x = Concatenate()([coords,x])

    x_gn, coords, nidx, dist = RaggedGravNet(n_neighbours=nneigh,
                                            n_dimensions=n_dimensions,
                                            n_filters=nfilt,
                                            n_propagate=nprop)([x, rs])
                                            
    x_mp = DistanceWeightedMessagePassing([64,64,32,32,16,16])([x,nidx,dist])
    
    x_pca = ApproxPCA()([coords,dist,x,nidx])
    x_pca = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x_pca)
    
    x = Concatenate()([x_pca,x_gn,x_mp])
    #check and compress it all
    x = Dense(128, activation='elu',name='dense_a_')(x)
    x = Dense(128, activation='elu',name='dense_b_')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
    
    x_append = Dense(nfilt//2, activation='elu',name='dense_c_')(x)
    
    energysums.append(MultiBackGather()([energy, gatherids]) )#assign energy sum to all cluster components

    allfeat.append(MultiBackGather()([x_append, gatherids]))

    backgatheredids.append(MultiBackGather()([sel_gidx, gatherids]))
    bgccoords = MultiBackGather()([ccoords, gatherids])
    backgathered_coords.append(bgccoords)

    rs = row_splits #important! here we are in non-reduced full graph mode again


    x = Concatenate(name='allconcat')(allfeat)
    x = Dense(128, activation='elu', name='alldense')(x)
    x = RaggedGlobalExchange()([x,row_splits])
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
    x = Dense(128, activation='elu')(x)
    x = GooeyBatchNorm(viscosity=viscosity, max_viscosity=max_viscosity,fluidity_decay=fluidity_decay)(x)
    x = Concatenate()([x]+energysums)
    x = Dense(64, activation='elu')(x)
    x = Dense(48, activation='elu')(x)
    x = Concatenate()([orig_coords,x,noise_score])#we have it anyway

    pred_beta, pred_ccoords, pred_dist, pred_energy,\
       pred_pos, pred_time, pred_id = create_outputs(x,feat,fix_distance_scale=False)

    #loss
    pred_beta = LLFullObjectCondensation(print_loss=True,
                                         energy_loss_weight=1e-2,
                                         position_loss_weight=1e-2,
                                         timing_loss_weight=1e-2,
                                         beta_loss_scale=1.,
                                         q_min=2.0,
                                         div_repulsion=True,
                                         # phase_transition=1,
                                         huber_energy_scale = 3,
                                         use_average_cc_pos=0.2,#smoothen it out a bit
                                         name="FullOCLoss"
                                         )(  #oc output and payload
                                            [pred_beta, pred_ccoords,pred_dist,
                                            pred_energy,pred_pos, pred_time, pred_id]+
                                             #truth information
                                            [orig_t_idx, orig_t_energy, 
                                             orig_t_pos, orig_t_time, orig_t_pid,
                                             orig_t_spectator_weight,
                                             row_splits])

    model_outputs = [('pred_beta', pred_beta), ('pred_ccoords',pred_ccoords),
       ('pred_energy',pred_energy),
       ('pred_pos',pred_pos),
       ('pred_time',pred_time),
       ('pred_id',pred_id),
       ('pred_dist',pred_dist),
       ('row_splits',row_splits)]

    for i, (x, y) in enumerate(zip(backgatheredids, backgathered_coords)):
        model_outputs.append(('backgatheredids_'+str(i), x))
        model_outputs.append(('backgathered_coords_'+str(i), y))
    return RobustModel(model_inputs=Inputs, model_outputs=model_outputs)

train = training_base_hgcal.HGCalTraining(testrun=False, resumeSilently=True, renewtokens=False)
train.val_data.writeToFile(train.outputDir + 'valsamples.djcdc')

if not train.modelSet():
    train.setModel(gravnet_model)
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(
        #clipnorm=0.001
        ))

    train.compileModel(learningrate=1e-4,
                       loss=None)



samplepath=train.val_data.getSamplePath(train.val_data.samples[0])


os.system('mkdir -p %s' % (train.outputDir + "/summary/"))
tensorboard_manager = TensorBoardManager(train.outputDir + "/summary/")

unique_id_path = os.path.join(train.outputDir,'unique_id.txt')
if os.path.exists(unique_id_path):
        with open(unique_id_path, 'r') as f:
            unique_id = f.readlines()[0].strip()

else:
    unique_id = str(uuid.uuid4())[:8]
    with open(unique_id_path, 'w') as f:
        f.write(unique_id+'\n')

nbatch = 30000


# This will both to server and a local file
database_manager = ExperimentDatabaseManager(file=os.path.join(train.outputDir,"training_metrics.db"), cache_size=100)
database_reading_manager = ExperimentDatabaseReadingManager(file=os.path.join(train.outputDir,"training_metrics.db"))
database_manager.set_experiment(unique_id)

metadata = matching_and_analysis.build_metadeta_dict(beta_threshold=0.5, distance_threshold=0.5, iou_threshold=0.0001, matching_type=matching_and_analysis.MATCHING_TYPE_MAX_FOUND)
analyzer = matching_and_analysis.OCAnlayzerWrapper(metadata)
optimizer = OCHyperParamOptimizer(analyzer=analyzer, limit_n_endcaps=10)
os.system('mkdir %s/full_validation_plots' % (train.outputDir))


learningrate = 1e-3
nbatch = 30000 #this is rather low, and can be set to a higher values e.g. when training on V100s

train.compileModel(learningrate=1e-3, #gets overwritten by CyclicLR callback anyway
                          loss=None,
                          metrics=None,
                          )

cb = []
model, history = train.trainModel(nepochs=1,
                                  run_eagerly=True,
                                  batchsize=nbatch,
                                  extend_truth_list_by = len(train.keras_model.outputs_keys)-2, #just adapt truth list to avoid keras error (no effect on model)
                                  batchsize_use_sum_of_squares=False,
                                  checkperiod=1,  # saves a checkpoint model every N epochs
                                  verbose=2,
                                  backup_after_batches=100,
                                  additional_callbacks=
                                  [CyclicLR (base_lr = learningrate/5,
                                  max_lr = learningrate,
                                  step_size = 150)]+cb)
