

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

'''
 outdict['predBeta']       = pred[:,:,0]
    outdict['predEnergy']     = pred[:,:,1]
    outdict['predEta']        = pred[:,:,2]
    outdict['predPhi']        = pred[:,:,3]
    outdict['predCCoords']    = pred[:,:,4:]
'''

def stupid_model(Inputs,feature_dropout=-1.):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    rs = Inputs[1]
    
    x = BatchNormalization(momentum=0.6)(x)
    
    print('x',x.shape)
    print('rs',rs.shape)
    
    #x = Dense(16,activation='tanh')(x)

    beta    = Dense(1, activation='sigmoid', name ="predBeta",trainable=True)(x)
    ener    = ScalarMultiply(10)(Dense(1, name ="predEnergy")(x))
    eta     = Dense(1, name ="predEta")(x)
    phi     = Dense(1, name ="predPhi")(x)
    ccoords = Dense(2, name ="predCCoords")(x)
    
    pred = Concatenate()([beta, ener, eta, phi, ccoords])
    
    return Model(inputs=Inputs, outputs=[pred, pred]) #explicit row split passing
    

train=training_base(testrun=False,resumeSilently=True,renewtokens=False)


from Losses import min_beta_loss_rowsplits, min_beta_loss_truth, pre_training_loss, null_loss

train.setModel(stupid_model,feature_dropout=-1)
    
train.compileModel(learningrate=1e-3,
                   loss=[min_beta_loss_truth,min_beta_loss_rowsplits],#fraction_loss)
                   clipnorm=1.) 
                  
print(train.keras_model.summary())

nbatch=8000 #this will be an upper limit on vertices per batch
verbosity=2

model,history = train.trainModel(nepochs=5, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity)



