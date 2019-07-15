


from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import  Dense,Conv1D, Conv2D, BatchNormalization, Multiply, Concatenate #etc
from Layers import GarNet, GravNet, GlobalExchange, CreateZeroMask
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training



def my_model(Inputs,nclasses,nregressions,otheroption):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    
    print('x',x.shape)
    orig_xyz = SelectFeatures(5,8)(x)
    
    mask = CreateZeroMask(0)(x)
    print(mask.shape)
    
    x = GlobalExchange()(x)
    for i in range(4):
        x = GlobalExchange()(x)
        x = Multiply()([x,mask])
        
        x = Dense(64,activation='elu')(x)
        x = Dense(48,activation='elu')(x)
        x = Dense(32,activation='tanh')(x)
        if i<3:
            x = GravNet(n_neighbours=24, n_dimensions=4, n_filters=48, n_propagate=12)(x)
        else:
            x, coords = GravNet(n_neighbours=24, n_dimensions=4, n_filters=48, n_propagate=12, also_coordinates=True)(x)
        x = Multiply()([x,mask])
        
        x = BatchNormalization()(x)
        
    
    x = Dense(32,activation='elu')(x)
    x = Dense(nregressions,activation=None)(x) #max 1 shower here
    x = Clip(-0.2, 1.2) (x)
    
    x = Concatenate()([x,orig_xyz,coords])
    print('pred[0] shape',x.shape)
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=True,renewtokens=True)




ppdt = plot_truth_pred_plus_coords_during_training(
               samplefile='/eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/ConverterTask/closeby_1.0To100.0_idsmix_dR0.1_n10_rnd1_s1/dev_LayerClusters_prod2/testconv/tuple_149_n100.meta',
               output_file=train.outputDir+'/train_progress',
               use_event=1,
               x_index = 5,
               y_index = 6,
               z_index = 7,
               pred_fraction_end = 20,
               transformed_x_index = 21,
               transformed_y_index = 22,
               transformed_z_index = 23,
               transformed_e_index = 24,
               cut_z='pos',
               afternbatches=1,
               on_epoch_end=False
               )


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
model,history = train.trainModel(nepochs=3, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=2,
                                 
                                 additional_callbacks=[ppdt.callback])

ppdt.end_job()
exit()

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





