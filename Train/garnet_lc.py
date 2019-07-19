

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
from Layers import GarNet, GravNet, GlobalExchange, CreateZeroMask
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training

n_garnet_layers=10
def garnet_model(Inputs,nclasses,nregressions,otheroption):
    
    nfilters=48
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    
    print('x',x.shape)
    
    mask = CreateZeroMask(0)(x)
    x = BatchNormalization(momentum=0.6)(x)
    x = Multiply()([x,mask])
    
    feats=[]
    for i in range(n_garnet_layers):
        
        x = GlobalExchange()(x)
        x = Multiply()([x,mask])
        x = Dense(nfilters,activation='tanh')(x)
        x = BatchNormalization(momentum=0.6)(x)
        x = Multiply()([x,mask])
        x = GarNet(n_aggregators=8, n_filters=nfilters, n_propagate=8, name = 'garnet_'+str(i),)(x)
        x = BatchNormalization(momentum=0.6)(x)
        x = Multiply()([x,mask])
        if n_garnet_layers <= 10 or i%2 == 0:
            feats.append(x)
        
    x = Concatenate()(feats)
    x = Dense(32,activation='tanh',name='pre_last_correction')(x)
    x = Dense(nregressions,activation=None,name='pre_last_correction_sm')(x)
    #x = SelectFeatures(1,nregressions+1)(x) #zero is the no-simcluster index
    #this cannot be used for the final trainings, but to get an idea of what is happening this might be useful
    
    #x = Clip(-0.5, 1.5) (x)
    
    
    #x = Concatenate()([x]+coords)
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=True,renewtokens=True)

sampledir = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/hgcalsim/CreateMLDataset/closeby_1.0To100.0_idsmix_dR0.2_n10_rnd1_s1/hitlist_layercluster/prod3'

#gets called every epoch
def decay_function(aftern_batches):
    return aftern_batches+5

ppdts=[ plot_truth_pred_plus_coords_during_training(
               samplefile=sampledir+'/tuple_9Of50_n100.meta',
               output_file=train.outputDir+'/train_progress'+str(i),
               use_event=5,
               x_index = 5,
               y_index = 6,
               z_index = 7,
               pred_fraction_end = 20,
               transformed_x_index = 20+4*i, #these will be ignored
               transformed_y_index = 21+4*i, #these will be ignored
               transformed_z_index = 22+4*i, #these will be ignored
               transformed_e_index = 23+4*i, #these will be ignored
               cut_z='pos',
               afternbatches=10,
               on_epoch_end=False,
               decay_function=decay_function,
               only_truth_and_pred=True
               ) for i in range(1) ]


ppdts_callbacks=[ppdts[i].callback for i in range(len(ppdts))]

from Losses import fraction_loss, fraction_loss_noweight

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(garnet_model,otheroption=1)
    
    #for regression use a different loss, e.g. mean_squared_error
train.compileModel(learningrate=0.003,
                   loss=fraction_loss,
                   clipnorm=1) 
                   
print(train.keras_model.summary())
#exit()
nbatch=300
verbosity=2
model,history = train.trainModel(nepochs=10, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 
                                 additional_callbacks=ppdts_callbacks)

nbatch=300
train.change_learning_rate(0.0005)
model,history = train.trainModel(nepochs=10+10, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 
                                 additional_callbacks=ppdts_callbacks)

train.change_learning_rate(0.0001)
#nbatch=nbatch*2
model,history = train.trainModel(nepochs=100+100, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 
                                 additional_callbacks=ppdts_callbacks)




train.change_learning_rate(0.00003)
model,history = train.trainModel(nepochs=100+100+100, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 
                                 additional_callbacks=ppdts_callbacks)


train.change_learning_rate(0.00001)
model,history = train.trainModel(nepochs=100+100+100+100, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 
                                 additional_callbacks=ppdts_callbacks)



for p in ppdts:
    p.end_job()
exit()


