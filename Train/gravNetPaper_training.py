
from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import  Conv1D, BatchNormalization, Concatenate #etc
from Layers import GravNet,GarNet, GlobalExchange


def gravnet(Inputs,nclasses,nregressions,otheroption=0):
    '''
    Just from the top of my head...
    '''
    
    nfilters=48
    nprop=22
    ndims=4
            
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    x = BatchNormalization(momentum=0.9)(x)
    feat = []
    for i in range(4):
        x = GlobalExchange()(x)
        x = Conv1D(64,1,activation='elu')(x)
        x = Conv1D(64,1,activation='elu')(x)
        x = Conv1D(64,1,activation='tanh')(x)
        x = GravNet(n_neighbours=40, n_dimensions=ndims, n_filters=nfilters, n_propagate=nprop, name='GravNet0_'+str(i))(x)
        x = BatchNormalization(momentum=0.9)(x)
        feat.append(x)
    
    x = Concatenate()(feat)
    x = Conv1D(128,1,activation='elu',name='last_conv_a')(x)
    x = Conv1D(3,1,activation='elu',name='last_conv_b')(x)
    x = Conv1D(2,1,activation='softmax',name="last_conv_output")(x) #that should be a 2
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)

def garnet(Inputs, bla,blu):
    
    x = Inputs[0]
    x = BatchNormalization(momentum=0.9)(x)
    
    for i in range(12):
        x = GarNet(n_aggregators=4, n_filters=48, n_propagate=12)(x)
    
    x = Conv1D(128,1,activation='relu',name='last_conv_a')(x)
    x = Conv1D(3,1,activation='relu',name='last_conv_b')(x)
    x = Conv1D(2,1,activation='softmax',name="last_conv_output")(x)
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


def dgcnn(Inputs, bla,blu):
    
    x = Inputs[0]
    
    feat = Conv1D(16,activation='relu')(x) #global transform to 3D
    x = BatchNormalization(momentum=0.9)(x)
        
    #    self.temp_feat_visualize = []
    #    feat = sparse_conv_edge_conv(feat,40,  [64,64,64])
    #    self.temp_feat_visualize.append(feat)
    #    feat_g = sparse_conv_global_exchange(feat)
    #
    #    feat = tf.layers.dense(tf.concat([feat,feat_g],axis=-1),
    #                           64, activation=tf.nn.relu )
    #
    #    feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum)
    #    if dropout>0:
    #        feat = tf.layers.dropout(feat, rate=dropout,training=self.is_train)
    #    
    #    feat1 = sparse_conv_edge_conv(feat,40, [64,64,64])
    #    self.temp_feat_visualize.append(feat1)
    #    feat1_g = sparse_conv_global_exchange(feat1)
    #    feat1 = tf.layers.dense(tf.concat([feat1,feat1_g],axis=-1),
    #                            64, activation=tf.nn.relu )
    #    feat1 = tf.layers.batch_normalization(feat1,training=self.is_train, momentum=self.momentum)
    #    if dropout>0:
    #        feat1 = tf.layers.dropout(feat1, rate=dropout,training=self.is_train)
    #    
    #    feat2 = sparse_conv_edge_conv(feat1,40,[64,64,64])
    #    self.temp_feat_visualize.append(feat2)
    #    feat2_g = sparse_conv_global_exchange(feat2)
    #    feat2 = tf.layers.dense(tf.concat([feat2,feat2_g],axis=-1),
    #                            64, activation=tf.nn.relu )
    #    feat2 = tf.layers.batch_normalization(feat2,training=self.is_train,momentum=self.momentum)
    #    if dropout>0:
    #        feat2 = tf.layers.dropout(feat2, rate=dropout,training=self.is_train)
    #    
    #    feat3 = sparse_conv_edge_conv(feat2,40,[64,64,64])
    #    self.temp_feat_visualize.append(feat3)
    #    feat3 = tf.layers.batch_normalization(feat3,training=self.is_train, momentum=self.momentum)
    #    if dropout>0:
    #        feat3 = tf.layers.dropout(feat3, rate=dropout,training=self.is_train)
    #    
    #    #global_feat = tf.layers.dense(feat2,1024,activation=tf.nn.relu)
    #    #global_feat = max_pool_on_last_dimensions(global_feat, skip_first_features=0, n_output_vertices=1)
    #    #print('global_feat',global_feat.shape)
    #    #global_feat = tf.tile(global_feat,[1,feat.shape[1],1])
    #    #print('global_feat',global_feat.shape)
    #    if useglobal:
    #        feat = tf.concat([feat,feat1,feat2,feat_g,feat1_g,feat2_g,feat3],axis=-1)
    #    else:
    #        feat = tf.concat([feat,feat1,feat2,feat3],axis=-1)
    #    feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
    #    feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
    #    return feat
    
    
    
from Losses import fraction_loss

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(gravnet)
    
    #for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=0.00003,
                   loss=fraction_loss) 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=10, 
                                 batchsize=10,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)




