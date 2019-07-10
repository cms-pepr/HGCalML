


from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import  Dense,Conv1D, Conv2D, BatchNormalization #etc
from Layers import GarNet, GravNet, GlobalExchange
from DeepJetCore.DJCLayers import ScalarMultiply, Clip

from tools import plot_pred_during_training



def my_model(Inputs,nclasses,nregressions,otheroption):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    
    print('x',x.shape)
    
    x = GlobalExchange()(x)
    for i in range(4):
        x = GlobalExchange()(x)
        x = Dense(64,activation='elu')(x)
        x = Dense(48,activation='elu')(x)
        x = Dense(32,activation='tanh')(x)
        x = GravNet(n_neighbours=24, n_dimensions=4, n_filters=48, n_propagate=12)(x)
        x = BatchNormalization()(x)
    
    x = Dense(32,activation='elu')(x)
    x = Dense(nregressions,activation=None)(x) #max 1 shower here
    x = Clip(-0.2, 1.2) (x)
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=True,renewtokens=True)



ppdt = plot_pred_during_training(
    samplefile='/eos/home-j/jkiesele/HGCal/HGCalML_data/training_data/conv/ntuple_converted_0_ntuple.meta',
               use_event=0,
               output_file=train.outputDir+'/train_progress',
               x_index = 5,
               y_index = 6,
               z_index = 7,
               e_index = 0,
               cut_z='pos',
               plotter=None,
               plotfunc=None)

ppdt.plotter.marker_scale=2.

from Losses import fraction_loss

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(my_model,otheroption=1)
    
    #for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=0.001,
                   loss=fraction_loss,
                   clipnorm=1) 
                   
print(train.keras_model.summary())

nbatch=2
model,history = train.trainModel(nepochs=100, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 
                                 additional_callbacks=[ppdt.callback])

train.change_learning_rate(0.0001)
model,history = train.trainModel(nepochs=100+100, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 
                                 additional_callbacks=[ppdt.callback])

train.change_learning_rate(0.00001)
model,history = train.trainModel(nepochs=100+100+100, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 
                                 additional_callbacks=[ppdt.callback])





