'''
Needs some cleaning.
On the longer term, let's keep this just a wrapper module for layers,
but the layers themselves to other files
'''

# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

from GraphCondensationLayers import graph_condensation_layers

global_layers_list.update(graph_condensation_layers)

# keras hacks

from tensorflow.keras.layers import LeakyReLU
global_layers_list['LeakyReLU'] = LeakyReLU


#base modules

from baseModules import PromptMetric
global_layers_list['PromptMetric'] = PromptMetric

from baseModules import LayerWithMetrics
global_layers_list['LayerWithMetrics'] = LayerWithMetrics

#metrics layers

from MetricsLayers import SimpleReductionMetrics
global_layers_list['SimpleReductionMetrics'] = SimpleReductionMetrics

from MetricsLayers import MLReductionMetrics
global_layers_list['MLReductionMetrics'] = MLReductionMetrics

from MetricsLayers import OCReductionMetrics
global_layers_list['OCReductionMetrics'] = OCReductionMetrics

#older layers

from LayersRagged import RaggedSumAndScatter
global_layers_list['RaggedSumAndScatter']=RaggedSumAndScatter

from LayersRagged import Condensate
global_layers_list['Condensate']=Condensate

from LayersRagged import CondensateToPseudoRS
global_layers_list['CondensateToPseudoRS']=CondensateToPseudoRS

from LayersRagged import GridMaxPoolReduction
global_layers_list['GridMaxPoolReduction']=GridMaxPoolReduction

from LayersRagged import RaggedGlobalExchange
global_layers_list['RaggedGlobalExchange']=RaggedGlobalExchange

##GravNet...


from GravNetLayersRagged import Abs
global_layers_list['Abs']=Abs

from GravNetLayersRagged import CastRowSplits
global_layers_list['CastRowSplits']=CastRowSplits

from GravNetLayersRagged import CreateMask
global_layers_list['CreateMask']=CreateMask

from GravNetLayersRagged import Where
global_layers_list['Where']=Where

from GravNetLayersRagged import MixWhere
global_layers_list['MixWhere']=MixWhere

from GravNetLayersRagged import ValAndSign
global_layers_list['ValAndSign']=ValAndSign

from GravNetLayersRagged import SplitOffTracks
global_layers_list['SplitOffTracks']=SplitOffTracks

from GravNetLayersRagged import MaskTracksAsNoise
global_layers_list['MaskTracksAsNoise']=MaskTracksAsNoise

from GravNetLayersRagged import ConcatRaggedTensors
global_layers_list['ConcatRaggedTensors']=ConcatRaggedTensors

from GravNetLayersRagged import CondensateToIdxs
global_layers_list['CondensateToIdxs']=CondensateToIdxs

from GravNetLayersRagged import CondensatesToPseudoRS
global_layers_list['CondensatesToPseudoRS']=CondensatesToPseudoRS

from GravNetLayersRagged import ReversePseudoRS
global_layers_list['ReversePseudoRS']=ReversePseudoRS

from GravNetLayersRagged import CleanCondensations
global_layers_list['CleanCondensations']=CleanCondensations

from GravNetLayersRagged import ScaleBackpropGradient
global_layers_list['ScaleBackpropGradient']=ScaleBackpropGradient

from GravNetLayersRagged import RemoveSelfRef
global_layers_list['RemoveSelfRef']=RemoveSelfRef

from GravNetLayersRagged import CreateIndexFromMajority
global_layers_list['CreateIndexFromMajority']=CreateIndexFromMajority

from GravNetLayersRagged import DownSample
global_layers_list['DownSample']=DownSample

from GravNetLayersRagged import PrintMeanAndStd
global_layers_list['PrintMeanAndStd']=PrintMeanAndStd

from GravNetLayersRagged import ElementScaling
global_layers_list['ElementScaling']=ElementScaling

from GravNetLayersRagged import ConditionalNormalizationLayer
global_layers_list['ConditionalNormalizationLayer']=ConditionalNormalizationLayer

from GravNetLayersRagged import ConditionalBatchNorm
global_layers_list['ConditionalBatchNorm']=ConditionalBatchNorm

from GravNetLayersRagged import ConditionalBatchEmbedding
global_layers_list['ConditionalBatchEmbedding']=ConditionalBatchEmbedding

from GravNetLayersRagged import GooeyBatchNorm
global_layers_list['GooeyBatchNorm']=GooeyBatchNorm

from GravNetLayersRagged import ScaledGooeyBatchNorm
global_layers_list['ScaledGooeyBatchNorm']=ScaledGooeyBatchNorm

from GravNetLayersRagged import ScaledGooeyBatchNorm2
global_layers_list['ScaledGooeyBatchNorm2']=ScaledGooeyBatchNorm2

from GravNetLayersRagged import SignedScaledGooeyBatchNorm
global_layers_list['SignedScaledGooeyBatchNorm']=SignedScaledGooeyBatchNorm

from GravNetLayersRagged import ConditionalScaledGooeyBatchNorm
global_layers_list['ConditionalScaledGooeyBatchNorm']=ConditionalScaledGooeyBatchNorm

from GravNetLayersRagged import ProcessFeatures
global_layers_list['ProcessFeatures']=ProcessFeatures

from GravNetLayersRagged import ManualCoordTransform
global_layers_list['ManualCoordTransform']=ManualCoordTransform

from GravNetLayersRagged import DirectedGraphBuilder
global_layers_list['DirectedGraphBuilder']=DirectedGraphBuilder

from GravNetLayersRagged import NeighbourCovariance
global_layers_list['NeighbourCovariance']=NeighbourCovariance

from GravNetLayersRagged import NeighbourApproxPCA
global_layers_list['NeighbourApproxPCA']=NeighbourApproxPCA

from GravNetLayersRagged import CreateGlobalIndices
global_layers_list['CreateGlobalIndices']=CreateGlobalIndices

from GravNetLayersRagged import LocalDistanceScaling
global_layers_list['LocalDistanceScaling']=LocalDistanceScaling

from GravNetLayersRagged import WeightedNeighbourMeans
global_layers_list['WeightedNeighbourMeans']=WeightedNeighbourMeans

from GravNetLayersRagged import WeightFeatures
global_layers_list['WeightFeatures']=WeightFeatures

from GravNetLayersRagged import RecalcDistances
global_layers_list['RecalcDistances']=RecalcDistances

from GravNetLayersRagged import SelectFromIndicesWithPad
global_layers_list['SelectFromIndicesWithPad']=SelectFromIndicesWithPad

from GravNetLayersRagged import SelectFromIndices
global_layers_list['SelectFromIndices']=SelectFromIndices

from GravNetLayersRagged import MultiBackGather
global_layers_list['MultiBackGather']=MultiBackGather

from GravNetLayersRagged import MultiBackScatter
global_layers_list['MultiBackScatter']=MultiBackScatter

from GravNetLayersRagged import MultiBackScatterOrGather
global_layers_list['MultiBackScatterOrGather']=MultiBackScatterOrGather

from GravNetLayersRagged import KNN
global_layers_list['KNN']=KNN

from GravNetLayersRagged import AddIdentity2D

global_layers_list['AddIdentity2D']=AddIdentity2D

from GravNetLayersRagged import WarpedSpaceKNN
global_layers_list['WarpedSpaceKNN']=WarpedSpaceKNN

from GravNetLayersRagged import SortAndSelectNeighbours
global_layers_list['SortAndSelectNeighbours']=SortAndSelectNeighbours

from GravNetLayersRagged import NoiseFilter
global_layers_list['NoiseFilter']=NoiseFilter

from GravNetLayersRagged import EdgeCreator
global_layers_list['EdgeCreator']=EdgeCreator

from GravNetLayersRagged import EdgeContractAndMix
global_layers_list['EdgeContractAndMix']=EdgeContractAndMix

from GravNetLayersRagged import EdgeSelector
global_layers_list['EdgeSelector']=EdgeSelector

from GravNetLayersRagged import DampenGradient
global_layers_list['DampenGradient']=DampenGradient

from GravNetLayersRagged import GroupScoreFromEdgeScores
global_layers_list['GroupScoreFromEdgeScores']=GroupScoreFromEdgeScores

from GravNetLayersRagged import NeighbourGroups
global_layers_list['NeighbourGroups']=NeighbourGroups

from GravNetLayersRagged import AccumulateNeighbours
global_layers_list['AccumulateNeighbours']=AccumulateNeighbours

from GravNetLayersRagged import SoftPixelCNN
global_layers_list['SoftPixelCNN']=SoftPixelCNN

from GravNetLayersRagged import RaggedGravNet
global_layers_list['RaggedGravNet']=RaggedGravNet

from GravNetLayersRagged import SelfAttention
global_layers_list['SelfAttention']=SelfAttention

from GravNetLayersRagged import MultiAttentionGravNetAdd
global_layers_list['MultiAttentionGravNetAdd']=MultiAttentionGravNetAdd

from GravNetLayersRagged import DynamicDistanceMessagePassing
global_layers_list['DynamicDistanceMessagePassing']=DynamicDistanceMessagePassing

from GravNetLayersRagged import CollectNeighbourAverageAndMax
global_layers_list['CollectNeighbourAverageAndMax']=CollectNeighbourAverageAndMax

from GravNetLayersRagged import MessagePassing
global_layers_list['MessagePassing']=MessagePassing

from GravNetLayersRagged import DistanceWeightedMessagePassing
global_layers_list['DistanceWeightedMessagePassing']=DistanceWeightedMessagePassing

from GravNetLayersRagged import ApproxPCA
global_layers_list['ApproxPCA']=ApproxPCA

from GravNetLayersRagged import DistanceWeightedAttentionMP
global_layers_list['DistanceWeightedAttentionMP']=DistanceWeightedAttentionMP

from GravNetLayersRagged import AttentionMP
global_layers_list['AttentionMP']=AttentionMP

from GravNetLayersRagged import EdgeConvStatic
global_layers_list['EdgeConvStatic']=EdgeConvStatic

from GravNetLayersRagged import XYZtoXYZPrime
global_layers_list['XYZtoXYZPrime']=XYZtoXYZPrime

from GravNetLayersRagged import SingleLocalGravNetAttention 
global_layers_list['SingleLocalGravNetAttention']=SingleLocalGravNetAttention 

from GravNetLayersRagged import LocalGravNetAttention 
global_layers_list['LocalGravNetAttention']=LocalGravNetAttention

### odd debug layers
from DebugLayers import PlotCoordinates
global_layers_list['PlotCoordinates']=PlotCoordinates


from DebugLayers import Plot2DCoordinatesPlusScore
global_layers_list['Plot2DCoordinatesPlusScore']=Plot2DCoordinatesPlusScore

from DebugLayers import PlotGraphCondensation
global_layers_list['PlotGraphCondensation']=PlotGraphCondensation


from DebugLayers import PlotEdgeDiscriminator
global_layers_list['PlotEdgeDiscriminator']=PlotEdgeDiscriminator

from DebugLayers import PlotNoiseDiscriminator
global_layers_list['PlotNoiseDiscriminator']=PlotNoiseDiscriminator


from DebugLayers import PlotGraphCondensationEfficiency
global_layers_list['PlotGraphCondensationEfficiency']=PlotGraphCondensationEfficiency


#ragged layers module
from RaggedLayers import ragged_layers
global_layers_list.update(ragged_layers)


from LossLayers import LLValuePenalty,LLNotNoiseClassifier,CreateTruthSpectatorWeights, NormaliseTruthIdxs, LLGraphCondOCLoss
from LossLayers import LLLocalClusterCoordinates, LLClusterCoordinates,LLFillSpace, LLOCThresholds
from LossLayers import LossLayerBase, LLBasicObjectCondensation, LLFullObjectCondensation,LLPFCondensates,LLNeighbourhoodClassifier
from LossLayers import LLEdgeClassifier, AmbiguousTruthToNoiseSpectator, LLGoodNeighbourHood, LLKnnPushPullObjectCondensation
from LossLayers import LLEnergySums,LLKnnSimpleObjectCondensation, LLPushTracks, LLFullOCThresholds, LLLocalEnergyConservation
import traceback
import os



##end debug


global_layers_list['AmbiguousTruthToNoiseSpectator']=AmbiguousTruthToNoiseSpectator
global_layers_list['NormaliseTruthIdxs']=NormaliseTruthIdxs

global_layers_list['CreateTruthSpectatorWeights']=CreateTruthSpectatorWeights

global_layers_list['LossLayerBase']=LossLayerBase
global_layers_list['LLNotNoiseClassifier']=LLNotNoiseClassifier
global_layers_list['LLValuePenalty']=LLValuePenalty

global_layers_list['LLPushTracks']=LLPushTracks
global_layers_list['LLEnergySums']=LLEnergySums


global_layers_list['LLOCThresholds']=LLOCThresholds
global_layers_list['LLLocalEnergyConservation']=LLLocalEnergyConservation
global_layers_list['LLFullOCThresholds']=LLFullOCThresholds
global_layers_list['LLFillSpace']=LLFillSpace
global_layers_list['LLClusterCoordinates']=LLClusterCoordinates
global_layers_list['LLLocalClusterCoordinates']=LLLocalClusterCoordinates
global_layers_list['LLKnnSimpleObjectCondensation']=LLKnnSimpleObjectCondensation
global_layers_list['LLKnnPushPullObjectCondensation']=LLKnnPushPullObjectCondensation
global_layers_list['LLBasicObjectCondensation']=LLBasicObjectCondensation
global_layers_list['LLFullObjectCondensation']=LLFullObjectCondensation
global_layers_list['LLGraphCondOCLoss']=LLGraphCondOCLoss
global_layers_list['LLPFCondensates']=LLPFCondensates
global_layers_list['LLNeighbourhoodClassifier']=LLNeighbourhoodClassifier
global_layers_list['LLEdgeClassifier']=LLEdgeClassifier
global_layers_list['LLGoodNeighbourHood']=LLGoodNeighbourHood



####### other stuff goes here
from Regularizers import OffDiagonalRegularizer,WarpRegularizer,AverageDistanceRegularizer,MeanMaxDistanceRegularizer

global_layers_list['OffDiagonalRegularizer']=OffDiagonalRegularizer
global_layers_list['WarpRegularizer']=WarpRegularizer
global_layers_list['AverageDistanceRegularizer']=AverageDistanceRegularizer
global_layers_list['MeanMaxDistanceRegularizer']=MeanMaxDistanceRegularizer




#also this one needs to be passed
from Initializers import EyeInitializer
global_layers_list['EyeInitializer']=EyeInitializer

####### some implementations



from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf






class SplitFeatures(Layer):
    def __init__(self,**kwargs):
        super(SplitFeatures, self).__init__(**kwargs)
        
    def call(self, inputs):
        n_f = inputs.shape[-1]
        return [inputs[...,i:i+1] for i in range(n_f)]

global_layers_list['SplitFeatures']=SplitFeatures

class GausActivation(Layer):
    def __init__(self,**kwargs):
        super(GausActivation, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, inputs):
        return tf.exp(-inputs**2)

global_layers_list['GausActivation']=GausActivation

class OnesLike(Layer):
    def __init__(self,**kwargs):
        super(OnesLike, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, inputs):
        return tf.ones_like(inputs)
    
global_layers_list['OnesLike']=OnesLike


class ZerosLike(Layer):
    def __init__(self,**kwargs):
        super(ZerosLike, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, inputs):
        return tf.zeros_like(inputs)
    
global_layers_list['ZerosLike']=ZerosLike

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
            self.model = tf.keras.Model().from_config(submodel, custom_objects=custom_objects)
        else:
            self.outputs_keys = [x[0] for x in model_outputs]
            outputs_placeholders = [x[1] for x in model_outputs]
            self.model = tf.keras.Model(inputs=model_inputs, outputs=outputs_placeholders)


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



#new implementation of RobustModel. Keep RobustModel for backwards-compat
class DictModel(tf.keras.Model):
    def __init__(self, 
                 inputs,
                 outputs: dict, #force to be dict
                 *args, **kwargs):
        """
        Just forces dictionary output
        """
        
        super(DictModel, self).__init__(inputs,outputs=outputs, *args, **kwargs)

    

global_layers_list['DictModel']=DictModel
