

###
#
#
# for testing: rm -rf TEST; python gravnet.py /eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/ConverterTask/closeby_1.0To100.0_idsmix_dR0.1_n10_rnd1_s1/dev_LayerClusters_prod2/testconv/dataCollection.dc TEST
#
###


from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import  Dense,Conv1D, Conv2D, BatchNormalization, Multiply, Concatenate #etc
from Layers import TransformCoordinates,AveragePoolVertices, GarNet, GravNet, GlobalExchange, CreateZeroMask, SortPredictionByEta, CenterPhi
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures, Print

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training
import tensorflow as tf

n_gravnet_layers=4 #+1
n_neighbours = 40

n_coords=n_gravnet_layers

isrechits = False
nbatch=50
passing_operations = 2
learningrate = 1e-4
if isrechits:
    nbatch=12*2
    passing_operations = 4
    learningrate = 1e-4
plots_after_n_batch=50
use_event=5

def norm_and_mask(x,mask):
    x = BatchNormalization(momentum=0.3)(x)
    x = Multiply()([x,mask])
    return x

def gravnet_model(Inputs,nclasses,nregressions,feature_dropout=-1.):
    coords=[]
    feats=[]
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    x = CenterPhi(2)(x)
    mask = CreateZeroMask(0)(x)
    

    etas_phis = SelectFeatures(1,3)(x)#eta, phi, just to propagate to the prediction
    r_coordinate = SelectFeatures(4,5)(x)
    energy = SelectFeatures(0,1)(x)
    x_all   = norm_and_mask(x,mask)
    x_basic = Concatenate()([etas_phis, r_coordinate, energy])#just for the kernel initializer
    x_basic = norm_and_mask(x_basic,mask)
    
    n_filters = 0
    for i in range(n_gravnet_layers):
        
        n_filters   = 32 + 24*i
        n_propagate =  8 + 8*i
        
        
        x = GlobalExchange()(x)
        
        x = Dense(n_filters,activation='elu')(x)
        x = norm_and_mask(x,mask)
        x = Dense(n_filters,activation='elu')(x)
        x = Dense(n_filters,activation='elu')(x)
        x = norm_and_mask(x,mask)
        x = Concatenate()([x_basic,x])
        x, coord = GravNet(n_neighbours=n_neighbours, 
                           n_dimensions=4, 
                           n_filters=n_filters, 
                           n_propagate=n_propagate, 
                           additional_message_passing=passing_operations,
                           name = 'gravnet_'+str(i),
                           also_coordinates=True,
                           feature_dropout=feature_dropout,
                           masked_coordinate_offset=-10)([x,mask])
        print(coord.shape)
        coords.append(coord)
        x = norm_and_mask(x,mask)
    
    x = Concatenate()([x_basic,x])
    x = Dense(n_filters,activation='elu')(x)
    x = Dense(n_filters,activation='elu')(x)
    x = norm_and_mask(x,mask)
    x = Dense(n_filters,activation='elu')(x)
    x = norm_and_mask(x,mask)
    
    
    x = Dense(nregressions,activation=None)(x) 
    x = norm_and_mask(x,mask)
    x = Concatenate()([x, x_all])
    x = Dense(64,activation='elu',name='last_correction')(x)
    x = Dense(nregressions,activation=None, kernel_initializer='zeros')(x) 
    x = Multiply()([x,mask])
    
    #x = Concatenate(name="concatlast", axis=-1)([x,coords])#+[n_showers]+[etas_phis])
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)



train=training_base(testrun=False,resumeSilently=True,renewtokens=True)

if isrechits:
    sampledir = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/hgcalsim/gnn.CreateMLDataset/closeby_1.0To100.0_idsmix_dR0.3_n5_rnd1_s1/hitlist/prod5'
    samplefile = sampledir+'/tuple_9Of250_n100.meta'
else:
    sampledir = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/hgcalsim/gnn.CreateMLDataset/closeby_1.0To100.0_idsmix_dR0.3_n5_rnd1_s1/hitlist_layercluster/prod5'
    samplefile = sampledir+'/tuple_9Of50_n100.meta'

#gets called every epoch
def decay_function(aftern_batches):
    return aftern_batches# int(aftern_batches+5)


ppdts= [ plot_truth_pred_plus_coords_during_training(
               samplefile=samplefile,
               output_file=train.outputDir+'/train_progress'+str(i+1),
               use_event=use_event,
               x_index = 5,
               y_index = 6,
               z_index = 7,
               e_index = 0,
               pred_fraction_end = 10,
               transformed_x_index = 10+4*i,
               transformed_y_index = 11+4*i,
               transformed_z_index = 12+4*i,
               transformed_e_index = 13+4*i,
               cut_z='pos',
               afternbatches=plots_after_n_batch,
               on_epoch_end=False,
               decay_function=decay_function
               ) for i in [0,n_coords-1] ] #only first and last

#dont use because of problems with TF keras?
ppdts_callbacks=None #[ppdts[i].callback for i in range(len(ppdts))]

from Losses import fraction_loss_with_penalties, fraction_loss_with_penalties_sort_pred
if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(gravnet_model,feature_dropout=-1)
    
    #read weights where possible from pretrained model
    #import os
    #from DeepJetCore.modeltools import load_model, apply_weights_where_possible
    #m_weights =load_model(os.environ['DEEPJETCORE_SUBPACKAGE'] + '/pretrained/gravnet_1.h5')
    #train.keras_model = apply_weights_where_possible(train.keras_model, m_weights)
    
    #for regression use a different loss, e.g. mean_squared_error
train.compileModel(learningrate=learningrate,
                   loss=fraction_loss_with_penalties,#fraction_loss)
                   clipnorm=10) 
                  
print(train.keras_model.summary())

verbosity=2

model,history = train.trainModel(nepochs=10, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)


train.change_learning_rate(learningrate/10.)

model,history = train.trainModel(nepochs=10+40, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)



train.change_learning_rate(learningrate/50.)
model,history = train.trainModel(nepochs=50+200, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)


train.change_learning_rate(learningrate/100.)
model,history = train.trainModel(nepochs=250+250, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)



for p in ppdts:
    p.end_job()
exit()


