

###
#
#
# for testing: rm -rf TEST; python gravnet.py /eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/ConverterTask/closeby_1.0To100.0_idsmix_dR0.1_n10_rnd1_s1/dev_LayerClusters_prod2/testconv/dataCollection.dc TEST
#
###


from DeepJetCore.training.training_base import training_base
from DeepJetCore.DataCollection import DataCollection
import keras
from keras.models import Model
from keras.layers import  Dense,Conv1D, Conv2D, BatchNormalization, Multiply, Concatenate #etc
from Layers import TransformCoordinates,AveragePoolVertices, GarNet, GravNet, GlobalExchange, CreateZeroMask, SortPredictionByEta, CenterPhi
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures, Print

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training
import tensorflow as tf
import os

n_gravnet_layers=3 #+1
n_coords=4

nbatch=1*15

plots_after_n_batch=50
use_event=5
learningrate=1e-5

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
    x_in = norm_and_mask(x,mask)
    
    etas_phis = SelectFeatures(1,3)(x)#eta, phi, just to propagate to the prediction
    r_coordinate = SelectFeatures(4,5)(x)
    energy = SelectFeatures(0,1)(x)
    x = Concatenate()([etas_phis, r_coordinate,x])#just for the kernel initializer
    
    x = norm_and_mask(x,mask)
    x, coord = GravNet(n_neighbours=40, n_dimensions=3, n_filters=80, n_propagate=16, 
                       name = 'gravnet_pre',
                       fix_coordinate_space=True,
                       also_coordinates=True,
                       masked_coordinate_offset=-10)([x,mask])
    x = norm_and_mask(x,mask)
    coords.append(coord)
    feats.append(x)
    
    
    for i in range(n_gravnet_layers):
        x = GlobalExchange()(x)
        
        x = Dense(64,activation='elu',name='dense_a_'+str(i))(x)
        x = norm_and_mask(x,mask)
        x = Dense(64,activation='elu',name='dense_b_'+str(i))(x)
        #x = Concatenate()([TransformCoordinates()(x),x])
        x = Dense(64,activation='elu',name='dense_c_'+str(i))(x)
        x = norm_and_mask(x,mask)
        x, coord = GravNet(n_neighbours=40, n_dimensions=4, n_filters=80, n_propagate=16, 
                           name = 'gravnet_'+str(i),
                           also_coordinates=True,
                           feature_dropout=feature_dropout,
                           masked_coordinate_offset=-10)([x,mask]) #shift+activation makes it impossible to mix real with zero-pad
        x = norm_and_mask(x,mask)
        coords.append(coord)
        feats.append(x)
    
    x = Concatenate()(feats)
    x = Dense(64,activation='elu',name='dense_a_last')(x)
    x = Dense(64,activation='elu',name='dense_b_last')(x)
    x = norm_and_mask(x,mask)
    x = Dense(64,activation='elu',name='dense_c_last')(x)
    x = norm_and_mask(x,mask)
    
    n_showers = AveragePoolVertices(keepdims=True)(x)
    n_showers = Dense(64,activation='elu',name='dense_n_showers_a')(n_showers)
    n_showers = Dense(1,activation=None,name='dense_n_showers')(n_showers)
    
    x = Dense(nregressions,activation=None,name='dense_pre_fracs')(x) 
    x = Concatenate()([x, x_in])
    x = Dense(64,activation='elu',name='dense_last_correction')(x)
    x = Dense(nregressions,activation=None,name='dense_fracs',
              kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x) 
    
    
    x = Concatenate(name="concatlast", axis=-1)([x]+coords+[n_showers]+[etas_phis])
    x = Multiply()([x,mask])
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)



train=training_base(testrun=False,resumeSilently=True,renewtokens=True)

plotdc=DataCollection(os.path.dirname(os.path.realpath(train.inputData))+'/merged_test.dc')

samplefile = plotdc.getSamplePath( plotdc.samples[0] )
#gets called every epoch
def decay_function(aftern_batches):
    return aftern_batches# int(aftern_batches+5)



ppdts=[ plot_truth_pred_plus_coords_during_training(
               samplefile=samplefile,
               output_file=train.outputDir+'/train_progress'+str(i+1),
               use_event=use_event,
               x_index = 5,
               y_index = 6,
               z_index = 7,
               e_index = 0,
               pred_fraction_end = 10,
               transformed_x_index = 13+4*i,
               transformed_y_index = 14+4*i,
               transformed_z_index = 15+4*i,
               transformed_e_index = 16+4*i,
               cut_z='pos',
               afternbatches=plots_after_n_batch,
               on_epoch_end=False,
               decay_function=decay_function
               ) for i in [n_gravnet_layers-1] ] #print only last
 

ppdts_callbacks=[ppdts[i].callback for i in range(len(ppdts))]

from Losses import fraction_loss_with_penalties_sort_pred,fraction_loss_with_penalties

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
                   loss=fraction_loss_with_penalties) 
                  
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


