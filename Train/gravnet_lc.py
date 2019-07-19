

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
from Layers import GarNet, GravNet, GlobalExchange, CreateZeroMask, SortPredictionByEta
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training

n_gravnet_layers=3 #+1
n_coords=4

def gravnet_model(Inputs,nclasses,nregressions,otheroption):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    
    print('x',x.shape)
    coords=[]
    
    mask = CreateZeroMask(0)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Multiply()([x,mask])
    x, coord = GravNet(n_neighbours=40, n_dimensions=4, n_filters=80, n_propagate=16, 
                       name = 'gravnet_pre',
                       also_coordinates=True)(x)
    coords.append(coord)
    x = BatchNormalization(momentum=0.9)(x)
    x = Multiply()([x,mask])
    
    feats=[]
    for i in range(n_gravnet_layers):
        x = GlobalExchange()(x)
        x = Multiply()([x,mask])
        x = Dense(64,activation='tanh')(x)
        x = Dense(64,activation='tanh')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Multiply()([x,mask])
        x = Dense(64,activation='sigmoid')(x)
        x = Multiply()([x,mask])
        x, coord = GravNet(n_neighbours=40, n_dimensions=4, n_filters=80, n_propagate=16, 
                           name = 'gravnet_'+str(i),
                           also_coordinates=True)(x)
        coords.append(coord)
        x = BatchNormalization(momentum=0.9)(x)
        x = Multiply()([x,mask])
        feats.append(x)
        
    x = Concatenate()(feats)
    x = Dense(64,activation='elu',name='pre_last_correction')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Multiply()([x,mask])
    x = Dense(nregressions,activation=None,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001))(x) 
    #x = Clip(-0.5, 1.5) (x)
    x = Multiply()([x,mask])
    
    x = SortPredictionByEta(input_energy_index=0, input_eta_index=1)([x,Inputs[0]])
    
    
    x = Concatenate()([x]+coords)
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)



train=training_base(testrun=False,resumeSilently=True,renewtokens=True)

sampledir = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/hgcalsim/CreateMLDataset/closeby_1.0To100.0_idsmix_dR0.2_n10_rnd1_s1/hitlist_layercluster/prod3'

#gets called every epoch
def decay_function(aftern_batches):
    return int(aftern_batches+5)

ppdts=[ plot_truth_pred_plus_coords_during_training(
               samplefile=sampledir+'/tuple_9Of50_n100.meta',
               output_file=train.outputDir+'/train_progress'+str(i),
               use_event=7,
               x_index = 5,
               y_index = 6,
               z_index = 7,
               e_index = 0,
               pred_fraction_end = 20,
               transformed_x_index = 20+4*i,
               transformed_y_index = 21+4*i,
               transformed_z_index = 22+4*i,
               transformed_e_index = 23+4*i,
               cut_z='pos',
               afternbatches=20,
               on_epoch_end=False,
               decay_function=decay_function
               ) for i in range(n_coords) ]


ppdts_callbacks=[ppdts[i].callback for i in range(len(ppdts))]

from Losses import fraction_loss, fraction_loss_noweight

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(gravnet_model,otheroption=1)
    
    #for regression use a different loss, e.g. mean_squared_error
train.compileModel(learningrate=0.001,
                   loss=fraction_loss)
                   #clipnorm=1) 
                   
print(train.keras_model.summary())

nbatch=2#100
verbosity=2

model,history = train.trainModel(nepochs=20, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 
                                 additional_callbacks=ppdts_callbacks)

train.change_learning_rate(0.0003)
model,history = train.trainModel(nepochs=50+20, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 
                                 additional_callbacks=ppdts_callbacks)



train.change_learning_rate(0.00003)
model,history = train.trainModel(nepochs=100+50+20, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)


train.change_learning_rate(0.00001)
model,history = train.trainModel(nepochs=100+50+20+100, 
                                 batchsize=nbatch,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)



for p in ppdts:
    p.end_job()
exit()


