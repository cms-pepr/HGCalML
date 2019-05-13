


from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten #etc

def my_model(Inputs,nclasses,nregressions,otheroption):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    x = Conv2D(4,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    
    x = Dense(nclasses)(x)
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)

def my_regression_model(Inputs,nclasses,nregressions,otheroption):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    x = Conv2D(4,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    
    x = Dense(nregressions, activaton='None')(x)
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)



train=training_base(testrun=False,resumeSilently=False,renewtokens=True)


if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(my_model,otheroption=1)
    
    #for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=0.003,
                   loss='categorical_crossentropy') 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=10, 
                                 batchsize=500,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)





