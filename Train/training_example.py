


from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import  Conv1D, Conv2D, BatchNormalization #etc
from Layers import GarNet

def my_model(Inputs,nclasses,nregressions,otheroption):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    
    print('x',x.shape)
    
    for i in range(10):
        x = BatchNormalization()(x)
        x = GarNet(n_aggregators=4, n_filters=32, n_propagate=16)(x)
    
    x = Conv1D(32,1,activation='relu')(x)
    x = Conv1D(7,1,activation='softmax')(x) #max 1 shower here
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=True)

from Losses import fraction_loss

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(my_model,otheroption=1)
    
    #for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=0.00003,
                   loss=fraction_loss) 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=10, 
                                 batchsize=1,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)





