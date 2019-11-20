

###
#
#
# for testing: rm -rf TEST; python gravnet.py /eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/ConverterTask/closeby_1.0To100.0_idsmix_dR0.1_n10_rnd1_s1/dev_LayerClusters_prod2/testconv/dataCollection.dc TEST
#
###


from DeepJetCore.training.training_base import training_base
import keras
from keras import Model
from keras.layers import  Dense,Conv1D, Conv2D, BatchNormalization, Multiply, Concatenate, Flatten
from Layers import TransformCoordinates,AveragePoolVertices, GarNet, GravNet, GlobalExchange, CreateZeroMask, SortPredictionByEta, CenterPhi
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures, Print

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training
import tensorflow as tf


def stupid_model(Inputs,feature_dropout=-1.):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    rs = Inputs[1]
    
    print('x',x.shape)
    print('rs',rs.shape)
    x = Concatenate()([x,rs ])
    x = Dense(1, name ="bla")(x)
    x = Flatten()(x)
    
    return Model(inputs=Inputs, outputs=[x])
    

train=training_base(testrun=False,resumeSilently=True,renewtokens=False)

def dumb_loss(truth, pred):
    print("called loss")
    print(pred.shape)
    print(truth.shape)
    return ( tf.reduce_mean(truth) - tf.reduce_mean(pred)  )**2

train.setModel(stupid_model,feature_dropout=-1)
    
train.compileModel(learningrate=1e-3,
                   loss=dumb_loss,#fraction_loss)
                   clipnorm=1) 
                  
print(train.keras_model.summary())

nbatch=10000 #this will be an upper limit on vertices per batch
verbosity=1

model,history = train.trainModel(nepochs=1, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity)



