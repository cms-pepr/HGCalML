

'''
How to use this script:

It still relies on DeepJetCore 1.X, so do NOT run a container or similar!
Just go to HGCalML from a clean environment.
source env.sh (NOT docker_env.sh)
And then you can run this script.

'''




import DeepJetCore
from DeepJetCore.TrainData import TrainData
from keras.models import load_model
from argparse import ArgumentParser
from plotting_tools import plotter_fraction_colors

#hacks to make the old layer work
from Losses import *
from Layers import *
from Metrics import *
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
import keras

######## JUST TO BE COMPATIBLE WITH THE OLD LAYERS USED IN THE MODEL. NO NEED TO FURTHER ADAPT THIS PART

from caloGraphNN import euclidean_squared, gauss, gauss_of_lin



class GravNetOld(keras.layers.Layer):
    def __init__(self, n_neighbours, n_dimensions, n_filters, n_propagate, name, 
                 also_coordinates=False, feature_dropout=-1, 
                 coordinate_kernel_initializer=keras.initializers.Orthogonal(),
                 other_kernel_initializer='glorot_uniform',
                 fix_coordinate_space=False, 
                 coordinate_activation=None,
                 masked_coordinate_offset=None,
                 additional_message_passing=2,
                 **kwargs):
        super(GravNetOld, self).__init__(**kwargs)
        print("created old layer")
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.name = name
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.masked_coordinate_offset = masked_coordinate_offset
        self.additional_message_passing = additional_message_passing
        
        self.input_feature_transform = keras.layers.Dense(n_propagate, name = name+'_FLR', kernel_initializer=other_kernel_initializer)
        self.input_spatial_transform = keras.layers.Dense(n_dimensions, name = name+'_S', kernel_initializer=coordinate_kernel_initializer, activation=coordinate_activation)
        self.output_feature_transform = keras.layers.Dense(n_filters, activation='tanh', name = name+'_Fout', kernel_initializer=other_kernel_initializer)

        self._sublayers = [self.input_feature_transform, self.input_spatial_transform, self.output_feature_transform]
        if fix_coordinate_space:
            self.input_spatial_transform = None
            self._sublayers = [self.input_feature_transform, self.output_feature_transform]
        
        self.message_passing_layers = []
        for i in range(additional_message_passing):
            self.message_passing_layers.append(
                keras.layers.Dense(n_propagate, activation='elu', 
                                   name = name+'_mp_'+str(i), kernel_initializer=other_kernel_initializer)
                )
        self._sublayers += self.message_passing_layers
        


    def build(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
            
        self.input_feature_transform.build(input_shape)
        if self.input_spatial_transform is not None:
            self.input_spatial_transform.build(input_shape)
        
        # tf.ragged FIXME?
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.input_feature_transform.units * 2))

        self.message_parsing_distance_weights=[]
        for i in range(len(self.message_passing_layers)):
            l = self.message_passing_layers[i]
            l.build((input_shape[0], input_shape[1], self.n_propagate*2))
            #d_weight = self.add_weight(name=self.name+'_mp_distance_weight_'+str(i), 
            #                          shape=(1,),
            #                          initializer='uniform',
            #                          constraint=keras.constraints.NonNeg() ,
            #                          trainable=True)
            #self.message_parsing_distance_weights.append(d_weight)

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)
        
        super(GravNetOld, self).build(input_shape)
        for l in self._sublayers:
            print(l.name, [a.shape for a in l.get_weights()])

    def call(self, x):
        
        mask = None
        if self.masked_coordinate_offset is not None:
            if not isinstance(x, list):
                raise Exception('GravNet: in mask mode, input must be list of input,mask')
            mask = x[1]
            x = x[0]
            
        
        if self.input_spatial_transform is not None:
            coordinates = self.input_spatial_transform(x)
        else:
            coordinates = x[:,:,0:self.n_dimensions]
            
        if self.masked_coordinate_offset is not None:
            sel_mask = tf.tile(mask, [1,1,tf.shape(coordinates)[2]])
            coordinates = tf.where(sel_mask>0., coordinates, tf.zeros_like(coordinates)-self.masked_coordinate_offset)

        collected_neighbours = self.collect_neighbours(coordinates, x, mask)

        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        output = self.output_feature_transform(updated_features)
        
        if self.masked_coordinate_offset is not None:
            output *= mask

        if self.also_coordinates:
            return [output, coordinates]
        return output
        

    def compute_output_shape(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
        if self.also_coordinates:
            return [(input_shape[0], input_shape[1], self.output_feature_transform.units),
                    (input_shape[0], input_shape[1], self.n_dimensions)]
        
        # tf.ragged FIXME? tf.shape() might do the trick already
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def collect_neighbours(self, coordinates, x, mask):
        
        # tf.ragged FIXME?
        # for euclidean_squared see caloGraphNN.py
        distance_matrix = euclidean_squared(coordinates, coordinates)
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]
        
        features = self.input_feature_transform(x)
        
        n_batches = tf.shape(features)[0]
        
        # tf.ragged FIXME? or could that work?
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]

        batch_range = tf.range(0, n_batches)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1) # (B, 1, 1, 1)

        # tf.ragged FIXME? n_vertices
        batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1]) # (B, V, N-1, 1)
        vertex_indices = tf.expand_dims(neighbour_indices, axis=3) # (B, V, N-1, 1)
        indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
    
        distance = -ranked_distances[:, :, 1:]
    
        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)
        
    
        for i in range(len(self.message_passing_layers)+1):
            if i:
                features = self.message_passing_layers[i-1](features)
                #w=self.message_parsing_distance_weights[i-1]
                #weights = gauss_of_lin(distance)
                #weights = tf.expand_dims(weights, axis=-1)
                
            if self.feature_dropout>0 and self.feature_dropout < 1:
                features = keras.layers.Dropout(self.feature_dropout)(features)
                
            neighbour_features = tf.gather_nd(features, indices) # (B, V, N-1, F)
            # weight the neighbour_features
            neighbour_features *= weights
            
            neighbours_max = tf.reduce_max(neighbour_features, axis=2)
            neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
            
            features = tf.concat([neighbours_max, neighbours_mean], axis=-1)
            if mask is not None:
                features *= mask
        
        return features

    def get_config(self):
            config = {'n_neighbours': self.n_neighbours, 
                      'n_dimensions': self.n_dimensions, 
                      'n_filters': self.n_filters, 
                      'n_propagate': self.n_propagate,
                      'name':self.name,
                      'also_coordinates': self.also_coordinates,
                      'feature_dropout' : self.feature_dropout,
                      'masked_coordinate_offset'       : self.masked_coordinate_offset,
                     # bug - was not saved 'additional_message_passing'    : self.additional_message_passing
                      }
            base_config = super(GravNetOld, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))











#overwrite

GravNet = GravNetOld
global_layers_list['GravNet']=GravNet
custom_objs = {}
custom_objs.update(djc_global_loss_list)
custom_objs.update(djc_global_layers_list)
custom_objs.update(global_loss_list)
custom_objs.update(global_layers_list)
custom_objs.update(global_metrics_list)

######## ######## ######## ######## ######## ACTUAL SCRIPT ######## ######## ######## 

parser = ArgumentParser('make plots pretty')
parser.add_argument('inputFile')
parser.add_argument('eventno', help="which event to plot")
parser.add_argument('outputFilePrefix')

args = parser.parse_args()

#this is part of the validation data!
sampledir = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/hgcalsim/gnn.CreateMLDataset/closeby_1.0To100.0_idsmix_dR0.3_n5_rnd1_s1/hitlist_layercluster/prod5'
samplefile = sampledir+'/tuple_9Of50_n100.meta'

modelfile = "/afs/cern.ch/user/j/jkiesele/public/HGCal_gravnet_lc2_multipass/KERAS_check_model_last.h5"


model=load_model(modelfile, custom_objects=custom_objs)

use_event=int(args.eventno)
x_index = 5
y_index = 6
z_index = 7
e_index = 0
pred_fraction_end = 20


def predict():
    '''
    returns features, predicted, truth
    '''
    td = TrainData()
    td.readIn(samplefile)
    td.skim(use_event)
    
    predicted = model.predict(td.x)
    if not isinstance(predicted, list):
        predicted=[predicted]
    return td.x, predicted, td.y


def makePlot(feat,predicted,truth, plot_truth=True):
    
    pred  = predicted[0][0] #list index, 0th event (skimmed before)
    truth = truth[0][0]
    feat  = feat[0][0]
    
    e = feat[:,e_index]
    z = feat[:,z_index]
    x = feat[:,x_index]
    y = feat[:,y_index]
    
    esel = e>0 #remove padded zeros
    
    pl = plotter_fraction_colors(output_file = "test",interactive = True) #you might want to change that
    truth_fracs = truth[:,0:-1]
    
    if plot_truth:
        pl.set_data(x[esel], y[esel], z[esel], e[esel], truth_fracs[esel])
    else:
        pl.set_data(x[esel], y[esel], z[esel], e[esel], pred_fracs[esel]) 
        
    pl.plot3d(ax=None) #this is a matplotlib ax object. Probably you'll need to create a figure and ax to add all the labels. Then add the ax object here to print into it
    pl.save_image()

f, p, t = predict()
makePlot(f, p, t, True)