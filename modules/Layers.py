

# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

from LayersRagged import *
from GravNetLayersRagged import ElementScaling,AddIdentity2D,WarpedSpaceKNN,GroupScoreFromEdgeScores,EdgeCreator,EdgeSelector,NoiseFilter,LNC,PrintMeanAndStd,GooeyBatchNorm,ManualCoordTransform,EdgeConvStatic,NeighbourApproxPCA,NormalizeInputShapes, NeighbourCovariance,LocalDistanceScaling,ProcessFeatures,GraphClusterReshape,SortAndSelectNeighbours,SoftPixelCNN, KNN, CollectNeighbourAverageAndMax, LocalClustering, CreateGlobalIndices, SelectFromIndices, MultiBackGather, RaggedGravNet, MessagePassing, DynamicDistanceMessagePassing, DistanceWeightedMessagePassing
from lossLayers import CreateTruthSpectatorWeights,LLLocalClusterCoordinates, LLClusterCoordinates, LossLayerBase, LLFullObjectCondensation
import traceback
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import os



global_layers_list['RaggedSumAndScatter']=RaggedSumAndScatter
global_layers_list['Condensate']=Condensate
global_layers_list['CondensateToPseudoRS']=CondensateToPseudoRS


global_layers_list['ElementScaling']=ElementScaling

global_layers_list['GroupScoreFromEdgeScores']=GroupScoreFromEdgeScores


global_layers_list['EdgeCreator']=EdgeCreator
global_layers_list['EdgeSelector']=EdgeSelector

global_layers_list['CreateTruthSpectatorWeights']=CreateTruthSpectatorWeights

global_layers_list['NoiseFilter']=NoiseFilter

global_layers_list['GridMaxPoolReduction']=GridMaxPoolReduction
global_layers_list['RaggedGlobalExchange']=RaggedGlobalExchange
global_layers_list['RaggedConstructTensor']=RaggedConstructTensor
global_layers_list['GraphShapeFilters']=GraphShapeFilters
global_layers_list['GraphFunctionFilters']=GraphFunctionFilters
global_layers_list['VertexScatterer']=VertexScatterer
global_layers_list['RaggedNeighborBuilder']=RaggedNeighborBuilder
global_layers_list['RaggedVertexEater']=RaggedVertexEater


global_layers_list['RaggedSelectThreshold']=RaggedSelectThreshold


global_layers_list['FusedRaggedGravNet']=FusedRaggedGravNet
global_layers_list['FusedRaggedGravNet_simple']=FusedRaggedGravNet_simple
global_layers_list['FusedMaskedRaggedGravNet']=FusedMaskedRaggedGravNet
global_layers_list['FusedRaggedGravNetLinParse']=FusedRaggedGravNetLinParse
global_layers_list['FusedRaggedGravNetLinParsePool']=FusedRaggedGravNetLinParsePool
global_layers_list['FusedRaggedGravNetGarNetLike']=FusedRaggedGravNetGarNetLike
global_layers_list['FusedRaggedGravNetAggAtt']=FusedRaggedGravNetAggAtt
global_layers_list['FusedRaggedGravNetDistMod']=FusedRaggedGravNetDistMod

global_layers_list['FusedRaggedGravNetRetDistLinParse']=FusedRaggedGravNetRetDistLinParse
global_layers_list['FusedRaggedGravNetRetDistDistMod']=FusedRaggedGravNetRetDistDistMod



global_layers_list['RaggedGravNet']=RaggedGravNet
global_layers_list['MessagePassing']=MessagePassing
global_layers_list['DynamicDistanceMessagePassing']=DynamicDistanceMessagePassing
global_layers_list['DistanceWeightedMessagePassing']=DistanceWeightedMessagePassing


global_layers_list['ProcessFeatures']=ProcessFeatures
global_layers_list['LocalDistanceScaling']=LocalDistanceScaling

global_layers_list['LocalClustering']=LocalClustering
global_layers_list['CreateGlobalIndices']=CreateGlobalIndices
global_layers_list['SelectFromIndices']=SelectFromIndices
global_layers_list['MultiBackGather']=MultiBackGather
global_layers_list['KNN']=KNN

global_layers_list['WarpedSpaceKNN']=WarpedSpaceKNN
global_layers_list['AddIdentity2D']=AddIdentity2D


global_layers_list['CollectNeighbourAverageAndMax']=CollectNeighbourAverageAndMax
global_layers_list['SoftPixelCNN']=SoftPixelCNN

global_layers_list['SortAndSelectNeighbours']=SortAndSelectNeighbours
global_layers_list['GraphClusterReshape']=GraphClusterReshape



global_layers_list['NeighbourCovariance']=NeighbourCovariance

global_layers_list['LLClusterCoordinates']=LLClusterCoordinates
global_layers_list['LLLocalClusterCoordinates']=LLLocalClusterCoordinates
global_layers_list['LLFullObjectCondensation']=LLFullObjectCondensation

global_layers_list['LossLayerBase']=LossLayerBase

global_layers_list['NormalizeInputShapes']=NormalizeInputShapes
global_layers_list['NeighbourApproxPCA']=NeighbourApproxPCA

global_layers_list['EdgeConvStatic']=EdgeConvStatic
global_layers_list['ManualCoordTransform']=ManualCoordTransform

global_layers_list['GooeyBatchNorm']=GooeyBatchNorm
global_layers_list['PrintMeanAndStd']=PrintMeanAndStd

global_layers_list['LNC']=LNC

####### other stuff goes here
from Regularizers import OffDiagonalRegularizer,WarpRegularizer,AverageDistanceRegularizer

global_layers_list['OffDiagonalRegularizer']=OffDiagonalRegularizer
global_layers_list['WarpRegularizer']=WarpRegularizer
global_layers_list['AverageDistanceRegularizer']=AverageDistanceRegularizer


#also this one needs to be passed
from initializers import EyeInitializer
global_layers_list['EyeInitializer']=EyeInitializer

####### some implementations



from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
from Loss_tools import deltaPhi


class OnesLike(Layer):
    def __init__(self,**kwargs):
        super(OnesLike, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, inputs):
        return tf.ones_like(inputs)
    
global_layers_list['OnesLike']=OnesLike

class CheckNaN(Layer):
    def __init__(self,**kwargs):
        super(CheckNaN, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, inputs):
        return tf.debugging.check_numerics(inputs, self.name+' detected NaNs or Infs')

global_layers_list['CheckNaN']=CheckNaN

class ReluPlusEps(Layer):
    def __init__(self,**kwargs):
        super(ReluPlusEps, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, inputs):
        return tf.nn.relu(inputs)+1e-6


global_layers_list['ReluPlusEps']=ReluPlusEps

class InputNormalization(Layer):
    def __init__(self,
                 multipliers,
                 biases,
                 **kwargs):
        super(ExpMinusOne, self).__init__(**kwargs)

        self.multiplier = tf.constant(multipliers, dtype='float32')
        self.multiplier = tf.expand_dims(self.multiplier, axis=0)
        self.bias = tf.constant(biases, dtype='float32')
        self.bias = tf.expand_dims(self.bias, axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return (inputs - self.bias) * self.multiplier


class ExpMinusOne(Layer):
    def __init__(self, **kwargs):
        super(ExpMinusOne, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return tf.math.expm1(inputs)


global_layers_list['ExpMinusOne']=ExpMinusOne


class CenterPhi(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour
    disappears for a small selection
    '''
    def __init__(self, phi_feature_index, **kwargs):
        super(CenterPhi, self).__init__(**kwargs)
        self.phi_feature_index=phi_feature_index

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        phi = inputs[...,self.phi_feature_index:self.phi_feature_index+1]

        reference = inputs[...,0:1,self.phi_feature_index:self.phi_feature_index+1]

        n_phi = deltaPhi( reference, phi )

        rest_left  = inputs[...,:self.phi_feature_index]
        rest_right = inputs[...,self.phi_feature_index+1:]

        return tf.concat( [rest_left, n_phi,rest_right ] , axis=-1 )

    def get_config(self):
        config = {'phi_feature_index': self.phi_feature_index}
        base_config = super(CenterPhi, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))


global_layers_list['CenterPhi']=CenterPhi


class CreateZeroMask(Layer):
    def __init__(self, feature_index, **kwargs):
        super(CreateZeroMask, self).__init__(**kwargs)
        self.feature_index=feature_index

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],1)

    def call(self, inputs):
        zeros = tf.zeros(shape=tf.shape(inputs)[:-1])
        mask = tf.where(inputs[:,:,self.feature_index]>0, zeros+1., zeros)
        mask = tf.expand_dims(mask,axis=2)
        return mask

    def get_config(self):
        config = {'feature_index': self.feature_index}
        base_config = super(CreateZeroMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))



global_layers_list['CreateZeroMask']=CreateZeroMask





class AveragePoolVertices(Layer):
    def __init__(self, keepdims=False, **kwargs):
        super(AveragePoolVertices, self).__init__(**kwargs)
        self.keepdims=keepdims

    def compute_output_shape(self, input_shape):
        if self.keepdims:
            return input_shape
        return (input_shape[0],input_shape[2])

    def call(self, inputs):
        #create mask
        n_nonmasked = tf.cast(tf.count_nonzero(tf.reduce_sum(inputs, axis=2), axis=-1), dtype='float32')
        n_nonmasked = tf.expand_dims(n_nonmasked, axis=1)
        rsum = tf.reduce_sum(inputs, axis=1)
        out = rsum/(n_nonmasked+K.epsilon())
        if self.keepdims:
            out = tf.expand_dims(out, axis=1)
            out = tf.tile(out, [1, tf.shape(inputs)[1], 1])
        return out

    def get_config(self):
        config = {'keepdims': self.keepdims}
        base_config = super(AveragePoolVertices, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))




global_layers_list['AveragePoolVertices']=AveragePoolVertices


class TransformCoordinates(Layer):
    def __init__(self, feature_index_x=0, feature_index_y=1, feature_index_z=2, **kwargs):
        super(TransformCoordinates, self).__init__(**kwargs)
        self.feature_index_x=feature_index_x
        self.feature_index_y=feature_index_y
        self.feature_index_z=feature_index_z

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],3)

    def call(self, inputs):

        xcoord = inputs[:,:,self.feature_index_x:self.feature_index_x+1]
        ycoord = inputs[:,:,self.feature_index_y:self.feature_index_y+1]
        zcoord = inputs[:,:,self.feature_index_z:self.feature_index_z+1]

        #transform to spherical coordinates
        r = tf.math.sqrt( xcoord**2 + ycoord**2 + zcoord**2 +K.epsilon())
        theta = tf.math.acos( zcoord / (r+K.epsilon()) )
        phi = tf.math.atan( ycoord / (xcoord+K.epsilon()) )

        ##replace nan values with 0 to deal with divergences
        #-> does not work for the gradient, needs to be fied before
        thetazeros = tf.zeros(shape=tf.shape(theta))
        phizeros = tf.zeros(shape=tf.shape(phi))
        theta = tf.where(tf.is_nan(theta), thetazeros, theta)
        phi = tf.where(tf.is_nan(phi), phizeros, phi)

        #replace cartesian coordinates with spherical ones calculated above
        #by concatenating tensors along last axis (the features axis)
        transf = tf.concat( [r,theta,phi], -1)

        return transf


    def get_config(self):
        config = {'feature_index_x': self.feature_index_x, 'feature_index_y': self.feature_index_y, 'feature_index_z': self.feature_index_z}
        base_config = super(TransformCoordinates, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))


global_layers_list['TransformCoordinates']=TransformCoordinates


class SortPredictionByEta(Layer):
    '''
    input: [predicted, input_features]
    arguments: input_energy_index (weight), input_eta_index (variable to be sorted by)
    output: sorted_predicted but
    '''
    def __init__(self, input_energy_index, input_eta_index , **kwargs):
        super(SortPredictionByEta, self).__init__(**kwargs)
        self.input_energy_index=input_energy_index
        self.input_eta_index=input_eta_index

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):

        predicted_fracs = inputs[0]
        energies        = inputs[1][:,:,self.input_energy_index:self.input_energy_index+1]
        etas            = inputs[1][:,:,self.input_eta_index:self.input_eta_index+1]
        pred_energies   = energies*predicted_fracs
        pred_sumenergy  = tf.reduce_sum(energies*predicted_fracs, axis=1)


        #predicted_fracs  : B x V x Fracs
        #energies         : B x V x 1
        #etas             : B x V x 1
        #pred_energies    : B x V x Fracs
        #pred_sumenergy   : B x Fracs

        weighted_etas = tf.reduce_sum(pred_energies * etas, axis=1)/(pred_sumenergy+K.epsilon())
        weighted_etas = tf.where(tf.abs(weighted_etas)>0.1, weighted_etas, tf.zeros_like(weighted_etas)+500.)
        # B x Fracs

        ranked_etas, ranked_indices = tf.nn.top_k(-weighted_etas, tf.shape(pred_sumenergy)[1])

        ranked_indices = tf.expand_dims(ranked_indices, axis=2)

        batch_range = tf.range(0, tf.shape(predicted_fracs)[0])
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)

        batch_indices = tf.tile(batch_range, [1, tf.shape(predicted_fracs)[2], 1]) # B x Fracs x 1
        indices = tf.concat([batch_indices, ranked_indices], axis=-1) # B x Fracs x 2

        identity_matrix = tf.eye(tf.shape(pred_sumenergy)[1]) #Fracs x Fracs 1 matrix
        identity_matrix = tf.expand_dims(identity_matrix, axis=0) # 1 x F x F
        identity_matrix = tf.tile(identity_matrix, [tf.shape(pred_sumenergy)[0],1,1])  # B x F x F
        sorted_identity_matrix = tf.gather_nd(identity_matrix, indices) # B x F x F

        # sorted_identity_matrix : B x Fm x Fm
        # predicted_fracs        : B x V  x Ff

        # B x Fm x Fm --> B x V x Fm x Fm
        sorted_identity_matrix = tf.expand_dims(sorted_identity_matrix, axis=1)
        sorted_identity_matrix = tf.tile(sorted_identity_matrix, [1,tf.shape(predicted_fracs)[1],1,1])
        # B x V x Fm x Fm

        # predicted_fracs   : B x V  x Ff --> B x V x Ff x 1
        sorted_predicted_fractions = tf.expand_dims(predicted_fracs, axis=3)

        return tf.squeeze(tf.matmul(sorted_identity_matrix, sorted_predicted_fractions), axis=-1)


    def get_config(self):
        config = {'input_energy_index': self.input_energy_index,
                  'input_eta_index': self.input_eta_index}
        base_config = super(SortPredictionByEta, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))




class Conv2DGlobalExchange(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(Conv2DGlobalExchange, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]+input_shape[3])

    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        average = tf.tile(average, [1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])
        return tf.concat([inputs,average],axis=-1)


    def get_config(self):
        base_config = super(Conv2DGlobalExchange, self).get_config()
        return dict(list(base_config.items()))


global_layers_list['Conv2DGlobalExchange']=Conv2DGlobalExchange


class ExtendedMetricsModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(ExtendedMetricsModel, self).__init__(*args, **kwargs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        ret_dict = {m.name: m.result() for m in self.metrics}

        ret_dict['x'] = x
        ret_dict['y_pred'] = y_pred

        return ret_dict


class RobustModel(tf.keras.Model):
    def __init__(self, skip_non_finite=5,
                 num_train_step=0,
                 model_inputs=None,
                 model_outputs=None,
                 custom_objects=None,
                 submodel=None,
                 *args, **kwargs):
        """

        :param skip_non_finite: Number of consecutive times to skip nans/inf loss and gradient values
        :param model_inputs: All the inputs to the model as a list
        :param model_outputs: All the outputs to the model as form of list of tupes where the first is the name of the output
        :param args: For subclass Model
        :param kwargs:  For subclass Model
        """

        super(RobustModel, self).__init__(*args, **kwargs)

        # if 'config' in kwargs:
        #     config = kwargs.pop('config')


        self.skip_non_finite = skip_non_finite
        self.non_finite_count = 0
        self.num_train_step = num_train_step # Since keras one is  not reliable (new epochs, if model is loaded from in between etc)

        if submodel is not None:
            self.outputs_keys = [x[0] for x in model_outputs]
            self.model = keras.Model().from_config(submodel, custom_objects=custom_objects)
        else:
            self.outputs_keys = [x[0] for x in model_outputs]
            outputs_placeholders = [x[1] for x in model_outputs]
            self.model = keras.Model(inputs=model_inputs, outputs=outputs_placeholders)


    def save(self, *args, **kwargs):
        if 'filepath' in kwargs:
            filepath = kwargs.pop('filepath')
        else:
            filepath = args[0]
            args = args[1:]

        if 'save_format' in kwargs:
            kwargs.pop('save_format')

        filepath_tf = filepath
        if str(filepath).endswith('.h5'):
            self.model.save(filepath=filepath)
            filepath_tf = os.path.splitext(filepath)[0] + '_save'

        save_format='tf'

        super().save(filepath=filepath_tf, save_format=save_format, save_traces=False)

    def call(self, inputs):
        outputs = self.model(inputs)

        return outputs

    def call_with_dict_as_output(self, inputs, numpy=False):
        outputs = self.call(inputs)
        output_keyed = {}
        for i in range(len(self.outputs_keys)):
            if numpy:
                output_keyed[self.outputs_keys[i]] = outputs[i].numpy()
            else:
                output_keyed[self.outputs_keys[i]] = outputs[i]

        return output_keyed

    def convert_output_to_dict(self, outputs):
        output_keyed = {}

        for i in range(len(self.outputs_keys)):
            output_keyed[self.outputs_keys[i]] = outputs[i]

        return output_keyed

    #def build(self, input_shape):
    #    super().build(input_shape)
    #    self.model.build(input_shape)

    def compile(self,
              *args,
              **kwargs):
        self.model.compile(*args, **kwargs)
        super().compile(*args, **kwargs)

    # def build(self, input_shape):
    #     self.model.build(input_shape)
    #     self.built=True

    def get_config(self):
        config = {'skip_non_finite': self.skip_non_finite, 'outputs_keys':self.outputs_keys}
        config['submodel'] = self.model.get_config()
        config['num_train_step'] = self.num_train_step
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        outputs_keys = config.pop('outputs_keys')
        outputs = [(x, None) for x in outputs_keys]

        xyz = cls(model_outputs=outputs, custom_objects=custom_objects, **config)
        return xyz

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data


        is_valid = True
        loss = None
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            try:
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            except Exception as e:
                is_valid = False
                traceback.print_stack()

        # Compute gradients
        trainable_vars = self.trainable_variables

        is_valid = is_valid and bool(tf.math.is_finite(loss))
        if is_valid:
            gradients = tape.gradient(loss, trainable_vars)
            is_valid = is_valid and not bool(tf.reduce_any([_grad is None for _grad in gradients]))

        if is_valid:
            num_non_finite_tensors = float(tf.reduce_sum(tf.cast([tf.reduce_any(tf.math.logical_not(tf.math.is_finite(_grad))) for _grad in gradients], tf.float32)))
            is_valid = is_valid and num_non_finite_tensors == 0.0

        if is_valid:
            self.non_finite_count = 0
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        else:
            if self.non_finite_count < self.skip_non_finite:
                print("\n\nWARNING: loss or gradient is not finite or error in loss. \nSkipping optimizer step %d/%d\n\n" % (self.non_finite_count+1, self.skip_non_finite))
            else:
                print("\n\nERROR: loss or gradient is not finite or error in loss. \nThrowing exception.\n\n")
                raise RuntimeError("Loss or gradient is not finite")

            self.non_finite_count += 1

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        ret_dict = {m.name: m.result() for m in self.metrics}

        self.data_x = x
        self.data_y_pred = y_pred
        self.num_train_step += 1

        return ret_dict


global_layers_list['ExtendedMetricsModel']=ExtendedMetricsModel
global_layers_list['RobustModel']=RobustModel
