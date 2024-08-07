import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Add, Multiply, BatchNormalization
from tensorflow.keras.layers import Flatten, Reshape

from DeepJetCore.DJCLayers import  SelectFeatures, ScalarMultiply, StopGradient
from GravNetLayersRagged import RaggedGravNet

from LossLayers import LLEnergySums, LLValuePenalty
from LossLayers import LLRegulariseGravNetSpace, LLEdgeClassifier
from LossLayers import LLClusterCoordinates, AmbiguousTruthToNoiseSpectator, LLNotNoiseClassifier
from LossLayers import LLFullOCThresholds, LLBasicObjectCondensation, LLFillSpace
from LossLayers import LLOCThresholds, LLKnnPushPullObjectCondensation, LLKnnSimpleObjectCondensation

from RaggedLayers import RaggedCollapseHitInfo, RaggedDense, RaggedToFlatRS, FlatRSToRagged, ForceExec
from RaggedLayers import RaggedCreateCondensatesIdxs, RaggedSelectFromIndices, RaggedMixHitAndCondInfo

from GravNetLayersRagged import MixWhere, ValAndSign, CollectNeighbourAverageAndMax
from GravNetLayersRagged import MultiBackScatterOrGather
from GravNetLayersRagged import RaggedGravNet
from GravNetLayersRagged import EdgeCreator, RandomSampling, MultiBackScatter
from GravNetLayersRagged import AccumulateNeighbours, SelectFromIndices, SelectFromIndicesWithPad
from GravNetLayersRagged import SortAndSelectNeighbours, NoiseFilter
from GravNetLayersRagged import CastRowSplits, ProcessFeatures
from GravNetLayersRagged import ScaledGooeyBatchNorm, GooeyBatchNorm, Where, MaskTracksAsNoise
from GravNetLayersRagged import ScaledGooeyBatchNorm2
from GravNetLayersRagged import Abs,RaggedGravNet,AttentionMP , LocalDistanceScaling, LocalGravNetAttention
from GravNetLayersRagged import NeighbourGroups, GroupScoreFromEdgeScores, ElementScaling, EdgeSelector, KNN
from GravNetLayersRagged import XYZtoXYZPrime, CreateMask, DistanceWeightedMessagePassing
from GravNetLayersRagged import SelfAttention, ScaledGooeyBatchNorm2, ConditionalScaledGooeyBatchNorm

from Layers import OnesLike, ZerosLike, Sqrt, CreateTruthSpectatorWeights
from Layers import RaggedGlobalExchange, CheckNaN, SphereActivation

from MetricsLayers import MLReductionMetrics, SimpleReductionMetrics, OCReductionMetrics
from DebugLayers import PlotCoordinates, PlotEdgeDiscriminator, PlotNoiseDiscriminator
from DebugLayers import PlotGraphCondensation, PlotGraphCondensationEfficiency
from GraphCondensationLayers import GraphCondensation, CreateGraphCondensation, PushUp, SelectUp, PullDown
from GraphCondensationLayers import LLGraphCondensationEdges, CreateGraphCondensationEdges, InsertEdgesIntoTransition
from GraphCondensationLayers import MLGraphCondensationMetrics, LLGraphCondensationScore, AddNeighbourDiff
from Initializers import EyeInitializer
from Regularizers import AverageDistanceRegularizer

from datastructures.TrainData_NanoML import n_id_classes
from oc_helper_ops import SelectWithDefault

from binned_select_knn_op import BinnedSelectKnn

from Layers import TranslationInvariantMP, DummyLayer



def random_sampling_block2(x, rs, gncoords, gnnidx, gndist, is_track, 
                           reduction=8., n_reduction=3, name='RSU',
                           activation = 'tanh'):
    """
    Unit that randomly samples vertices and backgatheres the information afterwards to the original shape
    In the reduced space, the vertices are connected by KNN and message passing is performed.

    Parameters
    ----------
    x : tf.Tensor
        Input features of shape (N, F)
    rs : tf.Tensor
        Row splits of shape (B,)
    gncoords : tf.Tensor
        Coordinates of shape (N, D)
    gnnidx : tf.Tensor
        Indices of shape (N, K)
    gndist : tf.Tensor
        Distances of shape (N, K)
    is_track : tf.Tensor
        Boolean tensor of shape (N,1)
    reduction : float, optional
        Reduction factor of each step, by default 8
    n_reduction : int, optional
        Number of reduction steps, by default 3
    layer_norm : bool
        Use `SphereActivation` instead of Batchnorm

    Example calculation with default parameters:
        If a large shower has 2048 hits and we we use 64 neighbours in the gravnet space,
        most vertices would only be connected within the shower.
        After two reduction steps we would typically have 32 vertices left, meaning that vertices
        are also connected to vertices from other showers. After the third reduction step on
        average only 4 hits would remain for a large shower and vertices would be connected
        over a large part of the calorimter.

        A full event with ~100k hits would be reduced to only 200 vertices after three reduction steps.
        Using 64 neighbours we can connect about 1/3 of the entire detector.
    """
    # D = tf.cast(gncoords.shape[-1], tf.int32)
    D = gncoords.shape[-1]
    # K = tf.cast(gnnidx.shape[-1], tf.int32)
    K = gnnidx.shape[-1]
    # N_Dense = 64
    # N_Dense = tf.cast(x.shape[-1], tf.int32)
    N_Dense = x.shape[-1]

    # x0 = Concatenate()([x, gncoords])
    # x_left0 = Dense(N_Dense, activation='elu', name=name + '_dense0')(x0)
    x_left0 = x

    xleft_list = [x_left0]  # left-sided features after message passing
    is_track_list = [is_track]
    row_splits_list = [rs]
    gncoords_list = [gncoords]
    gnnidx_list = [gnnidx]
    gndist_list = [gndist]
    size_list = []      # this list will stay one entry shorter than the others
    indices_list = []   # this list will stay one entry shorter than the others

    # Left side of the RSU
    for i in range(n_reduction):
        # Sampling and KNN
        # indices_selected_tmp are the indices regarding the input size
        x_temp, rs_temp, (size_tmp, indices_selected_tmp) = RandomSampling(
            reduction,
            name=name + f"_RSU_random_sampling{i}",
        )([xleft_list[-1], is_track_list[-1], row_splits_list[-1]])

        gravnetcoords_tmp = SelectFromIndices()(
            [indices_selected_tmp, gncoords_list[-1]])
        is_track_tmp = SelectFromIndices()(
            [indices_selected_tmp, is_track_list[-1]])
        gnnidx_tmp, gndist_tmp = KNN(
            K=K, record_metrics=True, 
            name=name + f"_RSU_KNN_{i}", min_bins=20)([gravnetcoords_tmp, rs_temp])
        gndist_tmp = gndist_tmp / (reduction**(2/D))
        gndist_tmp = AverageDistanceRegularizer(
                strength=1e-8,
                record_metrics=True,
                name=name+f"_dist_regularizer_{i}")(gndist_tmp)

        # Message Passing
        x_temp = Dense(N_Dense, activation=activation, name=name + f'_dense_left_preMP_{i}')(x_temp)
        x_temp = DistanceWeightedMessagePassing(
            name=name + f"_RSU_message_passing_left_{i}",
            n_feature_transformation = [64, 32, 32, 16],
            activation='elu',
        )([x_temp, gnnidx_tmp, gndist_tmp])
        x_temp = Dense(N_Dense, activation=activation, name=name + f'_dense_left_postMP_{i}')(x_temp)
        x_temp = ScaledGooeyBatchNorm2(fluidity_decay = 0.01,
                                       max_viscosity=0.9999,
                                       name=name+f'_gooey_left_{i}')(x_temp)
        
        # Bookkeeping
        xleft_list.append(x_temp)
        is_track_list.append(is_track_tmp)
        row_splits_list.append(rs_temp)
        gncoords_list.append(gravnetcoords_tmp)
        gnnidx_list.append(gnnidx_tmp)
        gndist_list.append(gndist_tmp)
        size_list.append(size_tmp)
        indices_list.append(indices_selected_tmp)

    # Right side of the RSU
    x_right_list = [xleft_list[-1]] # right-sided features after message passing
    for j in range(n_reduction):
        scatter_tmp = (size_list[-1-j], indices_list[-1-j]) # go backwards through the lists
        gnnidx_tmp = gnnidx_list[-2-j]  # We need to start with the second to last entry here
        gndist_tmp = gndist_list[-2-j]  # as the last entry is fully reduced and in this block 
        #                               # we start with the first 'back-scattered' level.

        x_temp = MultiBackScatter()([x_right_list[j], [scatter_tmp]])

        # Message Passing
        x_temp = Dense(N_Dense, activation=activation, name=name + f'_dense_right_preMP_{j}')(x_temp)
        # x_temp = CollectNeighbourAverageAndMax()([x_temp, gnnidx_tmp, gndist_tmp])
        x_temp = DistanceWeightedMessagePassing(
            name=name + f"_RSU_message_passing_right_{j}",
            n_feature_transformation = [64, 32, 32, 16],
            activation='elu',
        )([x_temp, gnnidx_tmp, gndist_tmp])
        x_temp = Dense(N_Dense, activation=activation, name=name + f'_dense_right_postMP_{j}')(x_temp)
        
        x_temp = ScaledGooeyBatchNorm2(fluidity_decay = 0.01,
                                       max_viscosity=0.9999,
                                       name=name+f'_gooey_right_{j}')(x_temp)
        
        # Skip connection
        x_temp = Add()([x_temp, xleft_list[-2-j]])

        x_right_list.append(x_temp)

    x_out = x_right_list[-1]
    return x_out


def random_sampling_block(x, rs, gncoords, gnnidx, gndist, is_track, reduction=8., n_reduction=3, layer_norm=False, name='RSU'):
    """
    Unit that randomly samples vertices and backgatheres the information afterwards to the original shape
    In the reduced space, the vertices are connected by KNN and message passing is performed.

    Parameters
    ----------
    x : tf.Tensor
        Input features of shape (N, F)
    rs : tf.Tensor
        Row splits of shape (B,)
    gncoords : tf.Tensor
        Coordinates of shape (N, D)
    gnnidx : tf.Tensor
        Indices of shape (N, K)
    gndist : tf.Tensor
        Distances of shape (N, K)
    is_track : tf.Tensor
        Boolean tensor of shape (N,1)
    reduction : float, optional
        Reduction factor of each step, by default 8
    n_reduction : int, optional
        Number of reduction steps, by default 3
    layer_norm : bool
        Use `SphereActivation` instead of Batchnorm

    Example calculation with default parameters:
        If a large shower has 2048 hits and we we use 64 neighbours in the gravnet space,
        most vertices would only be connected within the shower.
        After two reduction steps we would typically have 32 vertices left, meaning that vertices
        are also connected to vertices from other showers. After the third reduction step on
        average only 4 hits would remain for a large shower and vertices would be connected
        over a large part of the calorimter.

        A full event with ~100k hits would be reduced to only 200 vertices after three reduction steps.
        Using 64 neighbours we can connect about 1/3 of the entire detector.
    """
    # D = tf.cast(gncoords.shape[-1], tf.int32)
    D = gncoords.shape[-1]
    # K = tf.cast(gnnidx.shape[-1], tf.int32)
    K = gnnidx.shape[-1]
    # N_Dense = 64
    # N_Dense = tf.cast(x.shape[-1], tf.int32)
    N_Dense = x.shape[-1]

    # x0 = Concatenate()([x, gncoords])
    # x_left0 = Dense(N_Dense, activation='elu', name=name + '_dense0')(x0)
    x_left0 = x

    xleft_list = [x_left0]  # left-sided features after message passing
    row_splits_list = [rs]
    gncoords_list = [gncoords]
    gnnidx_list = [gnnidx]
    gndist_list = [gndist]
    is_track_list = [is_track]
    size_list = []      # this list will stay one entry shorter than the others
    indices_list = []   # this list will stay one entry shorter than the others

    # Left side of the RSU
    for i in range(n_reduction):
        # Sampling and KNN
        x_temp, rs_temp, (size_tmp, indices_selected_tmp) = RandomSampling(
            reduction,
            name=name + f"_RSU_random_sampling{i}",
        )([xleft_list[-1], is_track_list[-1], row_splits_list[-1]])

        gravnetcoords_tmp = SelectFromIndices()([indices_selected_tmp, gncoords])
        gnnidx_tmp, gndist_tmp = KNN(K=K, record_metrics=True, name=name + f"_RSU_KNN_{i}", min_bins=20)([gravnetcoords_tmp, rs_temp])
        gndist_tmp = gndist_tmp / (reduction**(2/D))    #TOOD: Double or triple check that this makes sense
        gndist_tmp = AverageDistanceRegularizer(
                strength=1e-8,
                record_metrics=True,
                name=name+f"_dist_regularizer_{i}")(gndist_tmp)

        # Message Passing
        x_temp = DistanceWeightedMessagePassing(
            name=name + f"_RSU_message_passing_left_{i}",
            n_feature_transformation = [64],
            activation='elu',
        )([x_temp, gnnidx_tmp, gndist_tmp])
        x_temp = Dense(N_Dense, activation='elu', name=name + f'_dense_left_postMP_{i}')(x_temp)

        if not layer_norm:
            x_temp = ScaledGooeyBatchNorm2(
                    fluidity_decay = 1e-1,
                    max_viscosity=0.1,
                    name=name+f'_gooey_left_{i}')(x_temp)
        else:
            x_temp = SphereActivation()(x_temp)

        # Bookkeeping
        xleft_list.append(x_temp)
        row_splits_list.append(rs_temp)
        gncoords_list.append(gravnetcoords_tmp)
        gnnidx_list.append(gnnidx_tmp)
        gndist_list.append(gndist_tmp)
        is_track_tmp = SelectFromIndices()([indices_selected_tmp, is_track_list[-1]])
        is_track_list.append(is_track_tmp)
        size_list.append(size_tmp)
        indices_list.append(indices_selected_tmp)

    # Right side of the RSU
    x_right_list = [xleft_list[-1]] # right-sided features after message passing
    for j in range(n_reduction):
        scatter_tmp = (size_list[-1-j], indices_list[-1-j]) # go backwards through the lists
        gnnidx_tmp = gnnidx_list[-2-j]  # We need to start with the second to last entry here
        gndist_tmp = gndist_list[-2-j]  # as the last entry is fully reduced and in this block 
        #                               # we start with the first 'back-scattered' level.

        x_temp = MultiBackScatter()([x_right_list[j], [scatter_tmp]])

        # Message Passing
        x_temp = DistanceWeightedMessagePassing(
            name=name + f"_RSU_message_passing_right_{j}",
            n_feature_transformation = [64],
            activation='elu',
        )([x_temp, gnnidx_tmp, gndist_tmp])
        x_temp = Dense(N_Dense, activation='elu', name=name + f'_dense_right_postMP_{j}')(x_temp)
        if not layer_norm:
            x_temp = ScaledGooeyBatchNorm2(
                    fluidity_decay = 1e-1,
                    max_viscosity=0.1,
                    name=name+f'_gooey_right_{j}')(x_temp)
        else:
            x_temp = SphereActivation()(x_temp)

        # Skip connection
        x_temp = Add()([x_temp, xleft_list[-2-j]])

        x_right_list.append(x_temp)

    x_out = x_right_list[-1]
    return x_out


def random_sampling_unit(x, rs, gncoords, gnnidx, gndist, is_track, reductions, gravnet_neighbours, gravnet_d, name='RSU'):
    """
    Unit that reduces the number of hits by random sampling in potentially multiple steps
    and backgatheres the information afterwards to the original shape
    """

    assert len(reductions) == 3
    assert isinstance(reductions[0], float)

    x_top_level = Concatenate()([x, gncoords])
    x_top_level = DistanceWeightedMessagePassing(
        name=name + f"_RSU_message_passing_toplevel_iteration",
        n_feature_transformation = [64],
        activation='elu',
    )([x, gnnidx, gndist])
    x_top_level = SphereActivation()(x_top_level)

    # First reduction block
    x_reduced_1, rs_reduced_1, (size_0, indices_selected_0) = RandomSampling(reductions[0])([x_top_level, is_track, rs])
    gravnetcoords_reduced_1 = SelectFromIndices()([indices_selected_0, gncoords])
    is_track_1 = SelectFromIndices()([indices_selected_0, is_track])
    gnnidx_reduced_1, gndist_reduced_1 = KNN(
        K=gravnet_neighbours, record_metrics=True,
        name=name + "_RSU_knn_reduction1", min_bins=20)([gravnetcoords_reduced_1, rs_reduced_1]) #TODO: the hardcoded part here might not be optimal

    gndist_reduced_1 = gndist_reduced_1 / (reductions[0]**gravnet_d) #TODO: verify this

    # Message passing on reduction level
    x_level_1 = x_reduced_1
    x_level_1 = DistanceWeightedMessagePassing(
        name=name + f"_RSU_message_passing_reduction1_iteration",
        n_feature_transformation = [64],
        activation='elu',
    )([x_level_1, gnnidx_reduced_1, gndist_reduced_1])
    x_level_1 = SphereActivation()(x_level_1)
    back_scattered_level1 = MultiBackScatter()([x_level_1, [(size_0, indices_selected_0)]])

    # Second reduction block
    x_reduced_2, rs_reduced_2, (size_1, indices_selected_1) = RandomSampling(reductions[1])([x_level_1, is_track_1, rs_reduced_1])
    gravnetcoords_reduced_2 = SelectFromIndices()([indices_selected_1, gncoords])
    is_track_2 = SelectFromIndices()([indices_selected_1, is_track_1])
    gnnidx_reduced_2, gndist_reduced_2 = KNN(
        K=gravnet_neighbours, record_metrics=True,
        name=name + "_RSU_knn_reduction2", min_bins=20)([gravnetcoords_reduced_2, rs_reduced_2]) #TODO: the hardcoded part here might not be optimal

    gndist_reduced_2 = gndist_reduced_2 / ((reductions[0] * reductions[1])**gravnet_d) #TODO: verify this

    # Message passing on reduction level
    x_level_2 = x_reduced_2
    x_level_2 = DistanceWeightedMessagePassing(
        name=name + f"_RSU_message_passing_reduction2_iteration",
        n_feature_transformation = [64],
        activation='elu',
    )([x_level_2, gnnidx_reduced_2, gndist_reduced_2])
    x_level_2 = SphereActivation()(x_level_2)
    back_scattered_level21 = MultiBackScatter()([x_level_2, [(size_1, indices_selected_1)]])
    back_scattered_level22 = MultiBackScatter()([back_scattered_level21, [(size_0, indices_selected_0)]])

    # Third reduction block
    x_reduced_3, rs_reduced_3, (size_2, indices_selected_2) = RandomSampling(reductions[2])([x_level_2, is_track_2, rs_reduced_2])
    gravnetcoords_reduced_3 = SelectFromIndices()([indices_selected_2, gncoords])
    gnnidx_reduced_3, gndist_reduced_3 = KNN(
        K=gravnet_neighbours, record_metrics=True,
        name=name + "_RSU_knn_reduction3", min_bins=20)([gravnetcoords_reduced_3, rs_reduced_3]) #TODO: the hardcoded part here might not be optimal

    gndist_reduced_3 = gndist_reduced_3 / ((reductions[0] * reductions[1] * reductions[2])**gravnet_d) #TODO: verify this

    # Message passing on reduction level
    x_level_3 = x_reduced_3
    x_level_3 = DistanceWeightedMessagePassing(
        name=name + f"_RSU_message_passing_reduction3_iteration",
        n_feature_transformation = [64],
        activation='elu',
    )([x_level_3, gnnidx_reduced_3, gndist_reduced_3])
    x_level_3 = SphereActivation()(x_level_3)
    back_scattered_level31 = MultiBackScatter()([x_level_3, [(size_2, indices_selected_2)]])
    back_scattered_level32 = MultiBackScatter()([back_scattered_level31, [(size_1, indices_selected_1)]])
    back_scattered_level33 = MultiBackScatter()([back_scattered_level32, [(size_0, indices_selected_0)]])

    x_out = Concatenate()([x_top_level, back_scattered_level1, back_scattered_level22, back_scattered_level33])

    return x_out


def extent_coords_if_needed(coords, x, n_cluster_space_coordinates,name='coord_extend'):
    if n_cluster_space_coordinates > 3:
        x = Concatenate()([coords,x])
        extendcoords = Dense(n_cluster_space_coordinates-3,
                             use_bias=False,
                             name=name+'_dense',
                             kernel_initializer=EyeInitializer(stddev=0.001)
                             )(x)
        coords = Concatenate()([coords, extendcoords])
    return coords

#new format!
def create_outputs(x, n_ccoords=3,
                   n_classes=n_id_classes,
                   n_pos = 2,
                   fix_distance_scale=True,
                   energy_factor=True,
                   name_prefix="output_module",
                   trainable=True,
                   is_track=None,
                   set_track_betas_to_one=False,
                   predict_spectator_weights=False,
                   pred_e_factor = 0.01):
    '''
    returns
        * pred_beta                     Dense(1)
        * pred_ccoords                  Dense(n_ccoords)
        * pred_dist                     1 if fix_distance_scale else 2*Dense(1)
        * pred_energy                   1 + Dense(1) if energy_factor else Dense(1)
        * pred_energy_low_quantile      Dense(1)
        * pred_energy_high_quantile     Dense(1)
        * pred_pos                      Dense(n_pos)
        * pred_time                     10 + Dense(1)
        * pred_time_unc                 1 + Dense(1)
        * pred_id                       Dense(n_classes)
        if predict_specator_weights:
            * pred_spectator_weights    Dense(1) with relu and L2 regularization
    '''
    if not fix_distance_scale:
        print("warning: fix_distance_scale=False can lead to issues.")

    pred_beta = Dense(1,
        activation='sigmoid',
        name = name_prefix+'_beta',
        trainable=trainable)(x)
    if set_track_betas_to_one:
        assert is_track is not None
        pred_beta = Where()([is_track, 0.9999, pred_beta])


    if predict_spectator_weights:
        pred_spectator_weights = Dense(1,
                activation='sigmoid',
                name = name_prefix+'_spectator_weight',
                trainable=trainable)(x)



    pred_ccoords = Dense(n_ccoords,
                         use_bias=False,
                         name = name_prefix+'_clustercoords',
                         trainable=trainable,
                         )(x) #bias has no effect


    energy_act=None
    if energy_factor:
        energy_act='elu'
    energy_res_act = 'relu'
    pred_energy = Dense(1,
        name = name_prefix+'_energy',
        kernel_initializer='zeros',
        activation=energy_act,
        trainable=trainable)(ScalarMultiply(pred_e_factor)(x))

    if energy_factor:
        pred_energy = Add(name= name_prefix+'_one_plus_energy')([OnesLike()(pred_energy),pred_energy])

    pred_energy_low_quantile = Dense(1,
        name = name_prefix+'_energy_low_quantile',
        # kernel_initializer='zeros',
        activation=energy_res_act,
        trainable=trainable)(x)

    pred_energy_high_quantile = Dense(1,
        name = name_prefix+'_energy_high_quantile',
        # kernel_initializer='zeros',
        activation=energy_res_act,
        trainable=trainable)(x)

    pred_pos =  Dense(n_pos,
        use_bias=False,
        name = name_prefix+'_pos',
        trainable=trainable)(x)

    pred_time = Dense(1,
        name=name_prefix + '_time_proxy',
        trainable=trainable)(ScalarMultiply(0.01)(x))
    pred_time = Add(
        name=name_prefix + '_time'
        )([ScalarMultiply(10.)(OnesLike()(pred_time)),pred_time])

    pred_time_unc = Dense(1,
        activation='elu',
        name = name_prefix+'_time_unc',
        trainable=trainable)(ScalarMultiply(0.01)(x))
    pred_time_unc = Add()([pred_time_unc, OnesLike()(pred_time_unc)])#strict positive with small turn on

    pred_id = Dense(n_classes,
        activation="softmax",
        name = name_prefix+'_class',
        trainable=trainable)(x)

    pred_dist = OnesLike()(pred_beta)
    if not fix_distance_scale:
        pred_dist = ScalarMultiply(2.)(Dense(1,
            activation='sigmoid',
            name = name_prefix+'_dist',
            trainable=trainable)(x))
        #this needs to be bound otherwise fully anti-correlated with coordates scale
    if predict_spectator_weights:
        return pred_beta, pred_ccoords, pred_dist, pred_energy, pred_energy_low_quantile, pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id, pred_spectator_weights
    else:
        return pred_beta, pred_ccoords, pred_dist, pred_energy, pred_energy_low_quantile, pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id




def re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_energy_low_quantile,
        pred_energy_high_quantile,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist,
        pred_pfc_idx = None,
        is_preselected_dataset=False,
        dict_output = True,
        ):

    assert dict_output #only dict output
    '''
    To be called after OC loss is applied to pre-selected outputs to bring it all back to the full dimensionality
    all hits that have been selected before and cannot be backgathered will be assigned coordinates far away,
    and a zero beta value.

    This is the counterpart of the pre_selection_model_full.

    returns full suite
    ('pred_beta', pred_beta),
    ('pred_ccoords', pred_ccoords),
    ('pred_energy_corr_factor', pred_energy_corr),
    ('pred_energy_low_quantile', pred_energy_low_quantile),
    ('pred_energy_high_quantile', pred_energy_high_quantile),
    ('pred_pos', pred_pos),
    ('pred_time', pred_time),
    ('pred_id', pred_id),
    ('pred_dist', pred_dist),
    ('row_splits', row_splits)
    '''
    from globals import cluster_space as  cs

    #this is only true if the preselection is run in situ - no preselected data set
    if 'scatterids' in pre_selection.keys():
        scatterids = pre_selection['scatterids']
        pred_ccoords = MultiBackScatterOrGather(default=cs.noise_coord)([pred_ccoords, scatterids])#set it far away for noise
        pred_beta = MultiBackScatterOrGather(default=0.)([pred_beta, scatterids])
        pred_energy_corr = MultiBackScatterOrGather(default=1.)([pred_energy_corr, scatterids])
        pred_energy_low_quantile = MultiBackScatterOrGather(default=1.)([pred_energy_low_quantile, scatterids])
        pred_energy_high_quantile = MultiBackScatterOrGather(default=1.)([pred_energy_high_quantile, scatterids])
        pred_pos = MultiBackScatterOrGather(default=0.)([pred_pos, scatterids])
        pred_time = MultiBackScatterOrGather(default=10.)([pred_time, scatterids])
        pred_id = MultiBackScatterOrGather(default=0.)([pred_id, scatterids])
        pred_dist = MultiBackScatterOrGather(default=1.)([pred_dist, scatterids])

        rechit_energy = ScalarMultiply(0.)(pred_beta) #FIXME, will duplicate.

    row_splits = None
    if is_preselected_dataset:
        row_splits = pre_selection['row_splits']
        rechit_energy = pre_selection['rechit_energy']
    else:
        row_splits = pre_selection['orig_row_splits']
        rechit_energy = pre_selection['rechit_energy'] #FIXME if not included here, return statement will fail, probably should be moved outside of if-statement

    ret_dict = {
            'pred_beta': pred_beta,
            'pred_ccoords': pred_ccoords,
            'pred_energy_corr_factor': pred_energy_corr,
            'pred_energy_low_quantile': pred_energy_low_quantile,
            'pred_energy_high_quantile': pred_energy_high_quantile,
            'pred_pos': pred_pos,
            'pred_time': pred_time,
            'pred_id': pred_id,
            'pred_dist': pred_dist,
            'rechit_energy': rechit_energy, #can also be summed if pre-selection
            'row_splits': row_splits }

    if pred_pfc_idx is not None:
        if 'scatterids' in pre_selection.keys():
            scatterids = pre_selection['scatterids']
            pred_pfc_idx = MultiBackScatterOrGather(default=-1)([pred_pfc_idx, scatterids])
        ret_dict.update({ 'pred_pfc_idx': pred_pfc_idx })

    return ret_dict




#also move this to the standard pre-selection  model



"""
def condition_input(orig_inputs, no_scaling=False):

    if not 't_spectator_weight' in orig_inputs.keys(): #compat layer
        orig_t_spectator_weight = CreateTruthSpectatorWeights(threshold=5.,minimum=1e-1,active=True
                                                         )([orig_inputs['t_spectator'],
                                                            orig_inputs['t_idx']])
        orig_inputs['t_spectator_weight'] = orig_t_spectator_weight


    if not 'is_track' in orig_inputs.keys():
        is_track = SelectFeatures(2,3)(orig_inputs['features'])
        orig_inputs['is_track'] =  Where(outputval=1.,condition='!=0')([is_track, ZerosLike()(is_track)])

    if not 'rechit_energy' in orig_inputs.keys():
        orig_inputs['rechit_energy'] = SelectFeatures(0, 1)(orig_inputs['features'])

    processed_features =  orig_inputs['features']
    orig_inputs['orig_features'] = orig_inputs['features']

    #get some things to work with
    orig_inputs['row_splits'] = CastRowSplits()(orig_inputs['row_splits'])
    orig_inputs['orig_row_splits'] = orig_inputs['row_splits']

    #coords have not been built so features not processed, so this is the first time this is called
    if not 'coords' in orig_inputs.keys():
        if not no_scaling:
            processed_features = ProcessFeatures(name='precondition_process_features')(orig_inputs['features'])

        orig_inputs['coords'] = SelectFeatures(5, 8)(processed_features)
        orig_inputs['features'] = processed_features

        #create starting point for cluster coords
        orig_inputs['prime_coords'] = XYZtoXYZPrime()(SelectFeatures(5, 8)(orig_inputs['orig_features']))

    return orig_inputs
"""


def condition_input(orig_inputs,
                    no_scaling=False,
                    no_prime=False,
                    new_prime=False):
    
    if not 't_spectator_weight' in orig_inputs.keys(): #compat layer
        orig_t_spectator_weight = CreateTruthSpectatorWeights(threshold=5.,minimum=1e-1,active=True
                                                         )([orig_inputs['t_spectator'],
                                                            orig_inputs['t_idx']])
        orig_inputs['t_spectator_weight'] = orig_t_spectator_weight


    if not 'is_track' in orig_inputs.keys():
        is_track = SelectFeatures(2,3)(orig_inputs['features'])
        orig_inputs['is_track'] =  Where(outputval=1.,condition='!=0')([is_track, ZerosLike()(is_track)])

    if not 'rechit_energy' in orig_inputs.keys():
        orig_inputs['rechit_energy'] = SelectFeatures(0, 1)(orig_inputs['features'])

    processed_features =  orig_inputs['features']
    orig_inputs['orig_features'] = orig_inputs['features']

    #get some things to work with
    orig_inputs['row_splits'] = CastRowSplits()(orig_inputs['row_splits'])
    orig_inputs['orig_row_splits'] = orig_inputs['row_splits']

    #coords have not been built so features not processed, so this is the first time this is called
    if not 'coords' in orig_inputs.keys():
        if not no_scaling:
            processed_features = ProcessFeatures(name='precondition_process_features')(orig_inputs['features'])

        orig_inputs['coords'] = SelectFeatures(5, 8)(processed_features)
        orig_inputs['features'] = processed_features

        #create starting point for cluster coords
        if no_prime:
            orig_inputs['prime_coords'] = SelectFeatures(5, 8)(orig_inputs['orig_features'])
        else:
            orig_inputs['prime_coords'] = XYZtoXYZPrime(new_prime=new_prime)(SelectFeatures(5, 8)(orig_inputs['orig_features']))

    return orig_inputs


def expand_coords_if_needed(coords, x, ndims, name, trainable):
    if coords.shape[-1] == ndims:
        return coords
    if coords.shape[-1] > ndims:
        raise ValueError("only expanding coordinates")
    return Concatenate()([ coords, Dense(ndims-coords.shape[-1],
                                         kernel_initializer='zeros',
                                         name=name,
                                         trainable=trainable)(x) ])



def mini_gravnet_block(K, x, coords, rs, trainable, record_metrics, name):

    x_in = x
    knncoords = ElementScaling(name=name+'gn_es1',trainable=trainable)(coords)
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn',
                    min_bins=20)([knncoords,rs])#hard code it here, this is optimised given our datasets

    x = Dense(64,activation='elu',name=name+'dense0',trainable=trainable)(x)
    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
                                       trainable=trainable,
                                       activation='elu')([x,nidx,dist])# hops are rather light

    return Concatenate()([x, x_in])

def mini_noise_block(sel, noise_threshold, trainable, record_metrics, name):

    isnotnoise = Dense(1, activation='sigmoid',trainable=trainable,
                           name=name+'_noisescore_d1',
                               )(sel['x'])



    isnotnoise = LLNotNoiseClassifier(active=trainable,record_metrics=record_metrics,
        scale=1.,
        purity_metrics_threshold = noise_threshold,
        record_efficiency=True
        )([isnotnoise, sel['t_idx']])

    isnotnoise = Where(outputval=1.,condition='!=0')([sel['is_track'], isnotnoise])
    sel['isnotnoise_score'] = isnotnoise


    no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(
        threshold = noise_threshold,
        record_metrics=record_metrics
        )([sel['isnotnoise_score'],sel['row_splits']])

    for k in sel.keys():
        sel[k] = SelectFromIndices()([no_noise_sel,sel[k]]) #also  good check, will fail if dimensions don't match

    sel['row_splits'] = no_noise_rs
    sel['noise_backscatter'] = noise_backscatter

    #this has removed hits, rebuild regular truth indices
    sel['t_idx'] = NormaliseTruthIdxs()([sel['t_idx'], sel['row_splits'] ])

    return sel


def mini_pre_condensation_model(inputs,
                           record_metrics=False,
                           trainable=False,
                           t_d=0.05,
                           t_b=0.9,
                           q_min=0.2,
                           purity_target=0.95,
                           print_batch_time=False,
                           name='pre_condensation',
                           condensation_mode = 'std',
                           noise_threshold=0.15,
                           cleaning_threshold=0.1,
                           cluster_dims=3,
                           condensate=True,
                           debug_outdir='',
                           debugplots_after=-1,
                           publishpath=None
                           ):

    K = 12

    orig_inputs = condition_input(inputs)

    #return {'coords': orig_inputs['prime_coords']}

    sel = orig_inputs.copy()
    sel['x'] = mini_gravnet_block(K, orig_inputs['features'],
                                  orig_inputs['prime_coords'],
                                  orig_inputs['row_splits'],
                                  trainable, record_metrics, name)

    ## REMOVE NOISE ##
    if noise_threshold > 0:
        sel = mini_noise_block(sel, noise_threshold, trainable, record_metrics, name)

    sel['x'] = Dense(64, activation='elu', name = name+'seld1')(sel['x'])
    ### object condensation part
    beta = Dense(1, activation='sigmoid',name=name+'dense_beta')(sel['x'])
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(sel['x']))

    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(sel['x'])
    ccoords = ScalarMultiply(0.01)(ccoords)

    prime_coords = ElementScaling(name=name+'es2',trainable=trainable)(sel['prime_coords'])
    prime_coords = expand_coords_if_needed(prime_coords, sel['x'],
                                             cluster_dims, name+'ccoords_exp',
                                             trainable=trainable)

    ccoords = Add()([prime_coords, ccoords])

    beta = LLBasicObjectCondensation(
           q_min=q_min,
           implementation = condensation_mode,
           print_batch_time = print_batch_time,
           record_batch_time = record_metrics,
           active=trainable,
           record_metrics = record_metrics,
           use_average_cc_pos=0.5
        )([beta, ccoords, d,sel['t_spectator_weight'],
                          sel['t_idx'], sel['row_splits']])

    out = sel
    out['orig_rowsplits'] = orig_inputs['row_splits']
    out['beta'] = beta
    out['features'] = Concatenate()([sel['features'],sel['x']])


    #track or masked
    #no_condensation_mask = tf.keras.layers.Maximum()([no_condensation_mask, sel['is_track']])
    no_condensation_mask = sel['is_track']

    ch_idx, c_idx, _, revflat, asso_idx, dyn_t_d, dyn_t_b = RaggedCreateCondensatesIdxs(t_d=t_d,t_b=t_b,
                                       active = condensate,
                                       keepnoise = True,
                                       return_thresholds = True,
                                       trainable = trainable,
                                       record_metrics = record_metrics,
                                       name = name+'_r_condensation')([beta, ccoords, d,
                                                                     no_condensation_mask,
                                                                     sel['row_splits']])

    out['cond_idx'] = OCReductionMetrics(name = name+'_metric',
        record_metrics=record_metrics
        )([asso_idx,  sel['t_idx']])

    if False:
        out['beta'] = LLOCThresholds(
                   name=name+'_ll_oc_thresholds',
                   active=trainable,
                   highest_t_d = 1.,
                   print_batch_time = print_batch_time,
                   record_batch_time = record_metrics,
                   lowest_t_b = 0.7,
                   purity_target = purity_target,
                   record_metrics=record_metrics)([beta, out['cond_idx'], d,
                                                   ccoords , sel['t_idx'], dyn_t_d, dyn_t_b])

    else:
        # betas, coords, d, c_idx, ch_idx, energy, t_idx, t_depe, t_d, t_b
        out['beta'] = LLFullOCThresholds(
            name=name+'_ll_full_oc_thresholds',
            active=trainable,
            purity_weight = 0.9,
            record_metrics=record_metrics,
             )([beta, ccoords, d, c_idx, ch_idx,
                sel['rechit_energy'], sel['t_idx'],
                sel['t_rec_energy'], dyn_t_d, dyn_t_b])


    if debugplots_after>0:
        out['ccoords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords',
                                        publish=publishpath)(
                                            [ccoords,
                                             beta,
                                             d,#additional features to add as hover data
                                             sel['t_idx'],sel['row_splits']])


        out['dummy'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords_cleaning',
                                        publish=publishpath)(
                                            [sel['coords'],
                                             sel['rechit_energy'],
                                             sel['t_idx'],#additional features to add as hover data
                                             asso_idx,sel['row_splits']])



    return out

def pre_condensation_model(inputs,
                           record_metrics=False,
                           trainable=False,
                           t_d=0.05,
                           t_b=0.9,
                           q_min=0.2,
                           purity_target=0.95,
                           print_batch_time=False,
                           name='pre_condensation',
                           condensation_mode = 'std',
                           noise_threshold=0.15,
                           cleaning_threshold=0.1,
                           cluster_dims=3,
                           condensate=True,
                           debug_outdir='',
                           debugplots_after=-1,
                           publishpath=None
                           ):

    K = 12

    orig_inputs = condition_input(inputs)

    coords = orig_inputs['prime_coords']
    energy = orig_inputs['rechit_energy']

    if debugplots_after>0:
        coords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_prime_coords',
                                        publish=publishpath)(
                                            [coords,
                                             energy,
                                             orig_inputs['t_idx'],orig_inputs['row_splits']])



    is_track = orig_inputs['is_track']
    x = orig_inputs['features']

    x = ScaledGooeyBatchNorm(record_metrics=record_metrics,
                             fluidity_decay = 1e-1, #very fast decay
                             trainable=trainable, name=name+'_gooey_0')(x)

    rs = orig_inputs['row_splits']

    ##### one simplified gravnet run

    coords = ElementScaling(name=name+'es1',trainable=trainable)(coords)
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn',
                    min_bins=20)([coords,rs])#hard code it here, this is optimised given our datasets

    ### this is the "meat" of the model

    x = Dense(32,activation='elu',name=name+'dense0',trainable=trainable)(x)

    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
                                       trainable=trainable,
                                       activation='elu')([x,nidx,dist])# hops are rather light

    x = ScaledGooeyBatchNorm(record_metrics=record_metrics,trainable=trainable,
                             fluidity_decay = 1e-2, #fast decay
                             name=name+'_gooey_1')(x)

    x_skip = x
    x = Dense(64,activation='elu',name=name+'dense1a',trainable=trainable)(x)
    x = Dense(64,activation='elu',name=name+'dense1b',trainable=trainable)(x)
    x = Dense(32,activation='elu',name=name+'dense1c',trainable=trainable)(x)
    x = Concatenate()([x, x_skip])


    ## for fair comparison reasons add something similar to the edge block in the standard pre-selection model

    dist,nidx = SortAndSelectNeighbours(K=-1,descending=True)([dist,nidx,energy])
    #create reduction, very simple, this gets expanded with K! be careful
    x_dist = StopGradient()(dist)
    x = Concatenate()([x,x_dist])
    x_e = Dense(8,activation='elu',name=name+'dense_x_e',trainable=trainable)(x)
    x_e = SelectFromIndicesWithPad()([nidx, x_e])#this will be big
    x_e = Flatten()(x_e)
    x_e = Dense(32,activation='elu',name=name+'dense_flat_x_e_0',trainable=trainable)(x_e)
    x_e = Dense(32,activation='elu',name=name+'dense_flat_x_e_1',trainable=trainable)(x_e)

    x_e = ScaledGooeyBatchNorm(record_metrics=record_metrics,
                             fluidity_decay = 1e-2, #fast decay
                             trainable=trainable , name=name+'_gooey_2')(x_e)

    x = Concatenate()([x, x_e]) #in this case just concat
    x = Dense(64, activation='elu',name=name+'dense_bef_lastmp',trainable=trainable)(x)

    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp2',
                                       trainable=trainable,
                                       activation='elu')([x,nidx,dist])
    x = Dense(64, activation='elu',name=name+'dense_last0',trainable=trainable)(x)
    x = Dense(64, activation='elu',name=name+'dense_last1',trainable=trainable)(x)

    x = ScaledGooeyBatchNorm(record_metrics=record_metrics,
                             fluidity_decay = 1e-2, #fast decay
                             trainable=trainable , name=name+'_gooey_3')(x)

    sel = orig_inputs.copy()
    sel['x'] = x

    ## REMOVE NOISE ##

    if noise_threshold > 0:

        isnotnoise = Dense(1, activation='sigmoid',trainable=trainable,
                           name=name+'_noisescore_d1',
                               )(sel['x'])

        isnotnoise = LLNotNoiseClassifier(active=trainable,record_metrics=record_metrics,
            scale=1.,
            purity_metrics_threshold = noise_threshold,
            record_efficiency=True
            )([isnotnoise, orig_inputs['t_idx']])

        #tracks are never noise here
        isnotnoise = Where(outputval=1.,condition='!=0')([is_track, isnotnoise])

        no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(threshold = noise_threshold,
                                                                   record_metrics=record_metrics
            )([isnotnoise,rs])


        for k in sel.keys():
            sel[k] = SelectFromIndices()([no_noise_sel,sel[k]]) #also  good check, will fail if dimensions don't match

        sel['row_splits'] = no_noise_rs

        # for later
        #scatterids = [noise_backscatter]


    ### object condensation part
    beta = Dense(1, activation='sigmoid',name=name+'dense_beta')(sel['x'])
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(sel['x']))

    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(sel['x'])
    selcoordsscaled = ElementScaling(name=name+'es2',trainable=trainable)(sel['coords'])
    exporigcoords = expand_coords_if_needed(selcoordsscaled,sel['x'],
                                             cluster_dims, name+'ccoords_exp',
                                             trainable=trainable)
    ccoords = Add()([exporigcoords, ccoords])


    #ccoords = LLFillSpace(active = trainable,
    #                       scale=0.1,
    #                       runevery=20,
    #                       record_metrics=record_metrics)([ccoords, sel['row_splits'],sel['t_idx']])

    #beta = LLBasicObjectCondensation(
    #    q_min=q_min,
    #    implementation = condensation_mode,
    #    print_batch_time = print_batch_time,
    #    record_batch_time = record_metrics,
    #    #scale = 0.1,
    #    active=trainable,
    #    record_metrics = record_metrics,
    #    use_average_cc_pos=0.
    #    )([beta, ccoords, d,sel['t_spectator_weight'],
    #                                    sel['t_idx'], sel['row_splits']])

    if condensation_mode == 'pushpull':
        beta = LLKnnPushPullObjectCondensation(q_min=q_min,
                                               mode='dippedsq',
                                           active=trainable,
                                           record_metrics = record_metrics)(
                                               [beta, ccoords, d, sel['t_idx'], sel['row_splits']])

    elif condensation_mode == 'simpleknn':
        beta = LLKnnSimpleObjectCondensation(active=trainable,
                                             name = name+'_simple_knn_oc',
                                           record_metrics = record_metrics)(
                                               [beta, ccoords, d, sel['t_idx'], sel['row_splits']]
                                               )

    else:
        beta = LLBasicObjectCondensation(
           q_min=q_min,
           implementation = condensation_mode,
           print_batch_time = print_batch_time,
           record_batch_time = record_metrics,
           #scale = 0.1,
           active=trainable,
           record_metrics = record_metrics,
           use_average_cc_pos=0.5
        )([beta, ccoords, d,sel['t_spectator_weight'],
                                        sel['t_idx'], sel['row_splits']])

    out = sel
    out['orig_rowsplits'] = orig_inputs['row_splits']
    out['beta'] = beta
    out['features'] = Concatenate()([sel['features'],sel['x']])

    #confscore = Dense(1,activation='sigmoid',name=name+'dense_conf')(sel['x'])

    #out['confscore'] = LLGoodNeighbourHood(
    #    active=trainable,
    #    distscale = t_d/4.,
    #    #scale=3.,#as it samples
    #    sampling = 0.03, #2k sampling for 100k batch
    #    record_metrics = record_metrics)([confscore, ccoords, d, sel['t_idx'], sel['row_splits']])

    #no_condensation_mask = CreateMask(threshold = cleaning_threshold,
    #                                   invert=True)(out['confscore'])

    #track or masked
    #no_condensation_mask = tf.keras.layers.Maximum()([no_condensation_mask, sel['is_track']])
    no_condensation_mask = sel['is_track']

    ch_idx, c_idx, _, revflat, asso_idx, dyn_t_d, dyn_t_b = RaggedCreateCondensatesIdxs(t_d=t_d,t_b=t_b,
                                       active = condensate,
                                       keepnoise = True,
                                       return_thresholds = True,
                                       trainable = trainable,
                                       record_metrics = record_metrics,
                                       name = name+'_r_condensation')([beta, ccoords, d,
                                                                     no_condensation_mask,
                                                                     sel['row_splits']])



    out['cond_idx'] = OCReductionMetrics(name = name+'_metric',
        record_metrics=record_metrics
        )([asso_idx,  sel['t_idx']])

    if False:
        beta = LLOCThresholds(
                   name=name+'_ll_oc_thresholds',
                   active=trainable,
                   highest_t_d = 1.,
                   print_batch_time = print_batch_time,
                   record_batch_time = record_metrics,
                   lowest_t_b = 0.7,
                   purity_target = purity_target,
                   record_metrics=record_metrics)([beta, out['cond_idx'], d,
                                                   ccoords , sel['t_idx'], dyn_t_d, dyn_t_b])

    else:
        # betas, coords, d, c_idx, ch_idx, energy, t_idx, t_depe, t_d, t_b
        beta = LLFullOCThresholds(
            name=name+'_ll_full_oc_thresholds',
            active=trainable,
            purity_weight = 0.9,
            record_metrics=record_metrics,
             )([beta, ccoords, d, c_idx, ch_idx,
                sel['rechit_energy'], sel['t_idx'],
                sel['t_rec_energy'], dyn_t_d, dyn_t_b])


    if debugplots_after>0:
        ccoords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords',
                                        publish=publishpath)(
                                            [ccoords,
                                             beta,
                                             d,#additional features to add as hover data
                                             sel['t_idx'],sel['row_splits']])


        out['dummy'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_ccoords_cleaning',
                                        publish=publishpath)(
                                            [sel['coords'],
                                             sel['rechit_energy'],
                                             sel['t_idx'],#additional features to add as hover data
                                             asso_idx,sel['row_splits']])

    out['ccoords'] = ccoords


    return out


#make this much simpler
def pre_selection_model(
        orig_inputs,
        debug_outdir='',
        trainable=False,
        name='pre_selection',
        debugplots_after=-1,
        reduction_threshold=0.8,#doesn't make a huge difference, this is higher purity than for layer clusters
        noise_threshold=0.1, #0.4 % false-positive, 96% noise removal
        K=12,
        record_metrics=True,
        filter_noise=True,
        double_knn=False,
        pass_through=False,
        pf_mode=False,
        activation='relu',
        pre_train_mode=False,
        print_time = False,
        ext_pf=1,
        hifp_penalty = 5,
        publish=None
        ):

    if pf_mode:
        activation='elu'
    else:
        ext_pf=0

    '''
    inputnames ['recHitFeatures', 'recHitFeatures_rowsplits', 't_idx', 't_idx_rowsplits', 't_energy', 't_energy_rowsplits', 't_pos', 't_pos_rowsplits',
    't_time', 't_time_rowsplits', 't_pid', 't_pid_rowsplits',
    't_spectator', 't_spectator_rowsplits', 't_fully_contained', 't_fully_contained_rowsplits',
    't_rec_energy', 't_rec_energy_rowsplits', 't_is_unique', 't_is_unique_rowsplits']

    ['t_idx', 't_energy', 't_pos', 't_time', 't_pid', 't_spectator', 't_fully_contained',
    't_rec_energy', 't_is_unique', 't_spectator_weight', 'coords', 'rechit_energy',
    'features', 'is_track', 'row_splits', 'scatterids', 'orig_row_splits']
    '''

    orig_inputs = condition_input(orig_inputs, no_scaling = pf_mode)

    if pass_through:
        orig_inputs['orig_row_splits'] = orig_inputs['row_splits']
        return orig_inputs

    rs = orig_inputs['row_splits']
    energy = orig_inputs['rechit_energy']
    coords = orig_inputs['coords']
    if pf_mode:
        coords = orig_inputs['prime_coords']
    is_track = orig_inputs['is_track']
    x = orig_inputs['features']
    if pf_mode:
        #quickly adjust
        x = ScaledGooeyBatchNorm(
            trainable=trainable,
                viscosity=0.01,
                fluidity_decay=1e-3,#gets to 1 rather quickly
                max_viscosity=1.)(x)
    x_skip = x

    #truth
    t_spec_w = orig_inputs['t_spectator_weight']
    t_idx = orig_inputs['t_idx']

    if pf_mode:
        coord_mult = Dense(32, activation='elu', name=name+'dcoord1',trainable=trainable)(coords)
        coord_mult = Dense(3, activation='sigmoid', name=name+'dcoord2',trainable=trainable)(coord_mult)
        coords = Multiply()([coords,coord_mult])

        if debugplots_after>0:
            coords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_pre_coords')(
                                            [coords,
                                             energy,
                                             t_idx,rs])

    coords = ElementScaling(name=name+'es1',trainable=trainable)(coords)

    coords = LLClusterCoordinates(active = trainable,
        name = name+'_LLClusterCoordinates',record_metrics = record_metrics,
        scale=1.
        )([coords,t_idx,t_spec_w,energy, rs])



    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn',
                    min_bins=20)([coords,rs])#hard code it here, this is optimised given our datasets

    ### this is the "meat" of the model

    x = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
                                       activation=activation,
                                       trainable=trainable)([x,nidx,dist])# hops are rather light
    x = Dense(32,activation=activation,name=name+'dense1a',trainable=trainable)(x)
    x = Dense(32,activation=activation,name=name+'dense1b',trainable=trainable)(x)



    #sort by energy, descending (not that it matters really)
    if not pf_mode:
        dist,nidx = SortAndSelectNeighbours(K=-1,descending=True)([dist,nidx,energy])
    else:
        dist,nidx = SortAndSelectNeighbours(K=-1)([dist,nidx])
    #create reduction, very simple, this gets expanded with K! be careful

    edge_complexity = [6,4]
    if pf_mode:
        edge_complexity = [12,12]
        if ext_pf and ext_pf == 1:
            edge_complexity = [24,16]
        if ext_pf and ext_pf == 2:
            edge_complexity = [24,24]
        x = Concatenate()([x,x_skip])

    x_e = Dense(edge_complexity[0],activation=activation,name=name+'dense_x_e',trainable=trainable)(x)
    #x_e = Concatenate()([is_track,x_e])
    x_e = SelectFromIndicesWithPad()([nidx, x_e])#this will be big
    x_e = Flatten()(x_e)
    x_e = Dense(edge_complexity[1]*K,activation=activation,name=name+'dense_flat_x_e',trainable=trainable)(x_e)

    if pf_mode:
        x_e = Dense(edge_complexity[1]*K,activation=activation,name=name+'dense_flat_x_e_2',trainable=trainable)(x_e)

    x_e = Reshape((K,edge_complexity[1]))(x_e)
    x_e = Dense(1,activation='sigmoid',name=name+'_ed3',trainable=trainable)(x_e)#edge classifier

    #not just where so that it ca be switched off without truth
    cluster_tidx = MaskTracksAsNoise(active=trainable)([t_idx,is_track])

    x_e = LLEdgeClassifier( name = name+'_LLEdgeClassifier',active=trainable,record_metrics=record_metrics,
            scale=5.,#high scale
            print_batch_time=print_time,
            fp_weight = 0.5, #0.9,
            hifp_penalty = hifp_penalty,
            lin_e_weight=True
            )([x_e,nidx,cluster_tidx, t_spec_w, orig_inputs['t_energy'], energy])



    if debugplots_after > 0:
        x_e = PlotEdgeDiscriminator(plot_every=debugplots_after,
                                    publish=publish,
                                        outdir=debug_outdir,name=name+'_edges')([x_e,nidx,cluster_tidx,orig_inputs['t_energy']])

    #skip the rest
    if pre_train_mode:
        return {'x_e': x_e}

    sel_nidx = EdgeSelector(
            threshold=reduction_threshold
            )([x_e,nidx])

    sel_t_spec_w, sel_t_idx = AmbiguousTruthToNoiseSpectator(
        record_metrics=record_metrics
        )([sel_nidx, t_spec_w, t_idx, energy])

    den_offset = 0.
    if pf_mode:
        den_offset = 12.
    hierarchy = GroupScoreFromEdgeScores(den_offset = den_offset)([x_e,sel_nidx])

    g_sel_nidx, g_sel, group_backgather, g_sel_rs = NeighbourGroups(threshold = 1e-3,return_backscatter=False,
        record_metrics = False)([hierarchy, sel_nidx, rs])


    g_sel = MLReductionMetrics(
        name=name+'_reduction_0',
        record_metrics = record_metrics
        )([g_sel,t_idx,orig_inputs['t_energy'],rs,g_sel_rs])

    #safety, these row splits are obsolete now
    rs = None

    # g_sel_sel_nidx: selected selected indices (first dimension output vertex multi)
    # g_sel_nidx: 'old' dimension, but with -1s where neighbours are clusters elsewhere
    g_sel_sel_nidx = SelectFromIndices()([g_sel,g_sel_nidx])

    #create selected output truth
    out={}
    for k in orig_inputs.keys():
        if 't_' == k[0:2]:
            out[k] = SelectFromIndices()([g_sel,orig_inputs[k]])

    #consider ambiguities
    out['t_idx'] = SelectFromIndices()([g_sel,sel_t_idx])
    out['t_spectator_weight'] = SelectFromIndices()([g_sel,sel_t_spec_w])

    ## create reduced features

    x_to_flat = Dense(4, activation=activation, name=name+'flatten_dense')(x)
    x_to_flat = Concatenate(name=name+'concat_x_to_flat')([x_skip,x_to_flat])
    #this has output dimension now
    x_flat_o = SelectFromIndicesWithPad()([g_sel_sel_nidx, x_to_flat])
    x_flat_o = Flatten()(x_flat_o)

    x_o = AccumulateNeighbours('minmeanmax')([x, g_sel_nidx, energy])
    x_o = SelectFromIndices()([g_sel,x_o])
    x_o = Concatenate(name=name+'concat_x_o')([x_o,x_flat_o])

    if pf_mode:
        x_o = ScaledGooeyBatchNorm(trainable=trainable)(x_o)
    else:
        x_o = GooeyBatchNorm(trainable=trainable)(x_o)

    #explicitly sum energy
    energy_o = AccumulateNeighbours('sum')([energy, g_sel_nidx])
    energy_o = SelectFromIndices()([g_sel,energy_o])

    #build preselection coordinates
    coords_o = AccumulateNeighbours('mean')([coords, g_sel_nidx, energy])
    coords_o = SelectFromIndices()([g_sel,coords_o])

    #pass original features
    mean_orig_feat = AccumulateNeighbours('mean')([orig_inputs['orig_features'], g_sel_nidx, energy])
    mean_orig_feat = SelectFromIndices()([g_sel,mean_orig_feat])

    ## selection done work on selected ones

    coord_add_o = Dense(16,activation=activation,name=name+'dense_coord_add1')(x_o)
    coord_add_o = Dense(3,name=name+'dense_coord_add3')(coord_add_o)
    coords_o = Add()([coords_o,coord_add_o])

    #was not clustered
    is_track_o = SelectFromIndices()([g_sel,is_track])

    #create a gradient for this
    coords_o = LLClusterCoordinates(
        record_metrics=record_metrics,
        record_batch_time=record_metrics,
        name = name+'_LLClusterCoordinates_coords_o',
        active = trainable,
        scale=1.
        )([coords_o,out['t_idx'],out['t_spectator_weight'],energy_o,g_sel_rs])

    #coords_o = LLFillSpace(active = trainable,
    #                       scale=0.1,
    #                       record_metrics=record_metrics)([coords_o,g_sel_rs,out['t_idx']])

    #add to dict:
    out['coords'] = coords_o
    out['rechit_energy'] = energy_o
    out['features'] = x_o
    out['orig_features'] = mean_orig_feat
    out['is_track'] = is_track_o

    if debugplots_after>0:
        out['coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_after_red')(
                                            [out['coords'],
                                             out['rechit_energy'],
                                             out['t_idx'],g_sel_rs])

    ####### noise filter part #########

    scatterids=None

    if filter_noise:

        isnotnoise = Dense(1, activation='sigmoid',trainable=trainable,name=name+'_noisescore_d1',
                           )(Concatenate(name=name+'concat_outf_outc')([out['features'],out['coords']]))

        #spectators are never noise here
        notnoisetruth = Where(outputval=1,condition='>0')([out['t_spectator_weight'], out['t_idx']])
        #tracks are never noise here
        notnoisetruth = Where(outputval=1,condition='>0')([out['is_track'], notnoisetruth])

        isnotnoise = LLNotNoiseClassifier(active=trainable,record_metrics=record_metrics,
            scale=1.)([isnotnoise, notnoisetruth])

        #tracks are never noise here**2
        isnotnoise = Where(outputval=1.,condition='>0')([out['is_track'], isnotnoise])

        if debugplots_after > 0:
            isnotnoise = PlotNoiseDiscriminator(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_noise_score')([isnotnoise,notnoisetruth])

        no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(threshold = noise_threshold,record_metrics=record_metrics
            )([isnotnoise,g_sel_rs])


        #select not-noise
        for k in out.keys():
            out[k] = SelectFromIndices()([no_noise_sel,out[k]]) #also  good check, will fail if dimensions don't match

        #safety, these row splits are obsolete now
        g_sel_rs = None

        out['row_splits'] = no_noise_rs
        scatterids = [group_backgather, noise_backscatter]

    else:
        out['row_splits'] = g_sel_rs
        scatterids = [group_backgather]

    if 'scatterids' in orig_inputs.keys():
        out['scatterids'] = orig_inputs['scatterids'] + scatterids
    else:
        out['scatterids'] = scatterids

    if not 'orig_row_splits' in orig_inputs.keys():
        out['orig_row_splits'] = orig_inputs['row_splits']
    else:
        out['orig_row_splits'] = orig_inputs['orig_row_splits']#pass through

    # full reduction metric
    out['row_splits'] = SimpleReductionMetrics(
        name=name+'full_reduction',
        record_metrics=record_metrics
        )([out['row_splits'],orig_inputs['row_splits']])


    beta = Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_beta')(out['features'])
    d = ScalarMultiply(2.)(Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_d')(out['features']))
    if trainable:
        beta = LLBasicObjectCondensation(
            scale=0.1,
            active=trainable,
            record_batch_time = record_metrics,
            record_metrics = record_metrics,
            use_average_cc_pos=0.1,
            name=name+'_basic_object_condensation'
            )([beta, out['coords'], d,out['t_spectator_weight'],out['t_idx'], out['row_splits']])

    out['features'] = Concatenate()([out['features'],d,beta])

    return out



def noise_filter_block(orig_inputs, x, name, trainable,
                       record_metrics, noise_threshold, rs,
                       print_time=False,
                       debugplots_after=-1,
                       debug_outdir=None,
                       publish = None):

    #orig_inputs.update({'x':x})
    #return orig_inputs

    isnotnoise = Dense(1, activation='sigmoid',trainable=trainable,name=name+'_noisescore_d1',
                       )(x)

    #spectators are never noise here
    notnoisetruth = Where(outputval=1,condition='>0')([orig_inputs['t_spectator_weight'], orig_inputs['t_idx']])
    notnoisetruth = Where(outputval=1,condition='>0')([orig_inputs['is_track'], notnoisetruth])

    isnotnoise = LLNotNoiseClassifier(active=trainable,record_metrics=record_metrics,
                                      print_time=print_time,
        scale=1.)([isnotnoise, notnoisetruth])

    #tracks are never noise here**2
    isnotnoise = Where(outputval=1.,condition='>0')([orig_inputs['is_track'], isnotnoise])

    if debugplots_after > 0:
        isnotnoise  = PlotNoiseDiscriminator(
                name= name+'_noise',
                plot_every=debugplots_after,
                outdir=debug_outdir,
                publish=publish)([isnotnoise, notnoisetruth])

    no_noise_sel, no_noise_rs, noise_backscatter = NoiseFilter(threshold = noise_threshold,
                                                               #print_reduction=True,
                                                               record_metrics=record_metrics
        )([isnotnoise,rs])

    out = orig_inputs
    out['x'] = x
    #select not-noise
    for k in out.keys():
        out[k] = SelectFromIndices()([no_noise_sel,out[k]]) #also  good check, will fail if dimensions don't match


    out['row_splits'] = no_noise_rs
    scatterids = noise_backscatter

    if 'scatterids' in orig_inputs.keys():
        out['scatterids'] = orig_inputs['scatterids'] + scatterids
    else:
        out['scatterids'] = scatterids

    if not 'orig_row_splits' in orig_inputs.keys():
        out['orig_row_splits'] = orig_inputs['row_splits']
    else:
        out['orig_row_splits'] = orig_inputs['orig_row_splits']#pass through

    return out

#make this much simpler
def pre_selection_model2(
        orig_inputs,
        debug_outdir='',
        trainable=False,
        name='pre_selection2',
        debugplots_after=-1,
        reduction_threshold=0.8,
        noise_threshold=0.4, #0.4 % false-positive, 96% noise removal
        K=20,
        record_metrics=True,
        pass_through=False,
        pre_train_mode=False,
        print_time = False,
        hifp_penalty = 5,
        publish=None,
        flat_edges = False,
        lin_edge_weight=False,
        fill_space_loss=0.
        ):

    activation='elu'

    '''
    inputnames ['recHitFeatures', 'recHitFeatures_rowsplits', 't_idx', 't_idx_rowsplits', 't_energy', 't_energy_rowsplits', 't_pos', 't_pos_rowsplits',
    't_time', 't_time_rowsplits', 't_pid', 't_pid_rowsplits',
    't_spectator', 't_spectator_rowsplits', 't_fully_contained', 't_fully_contained_rowsplits',
    't_rec_energy', 't_rec_energy_rowsplits', 't_is_unique', 't_is_unique_rowsplits']

    ['t_idx', 't_energy', 't_pos', 't_time', 't_pid', 't_spectator', 't_fully_contained',
    't_rec_energy', 't_is_unique', 't_spectator_weight', 'coords', 'rechit_energy',
    'features', 'is_track', 'row_splits', 'scatterids', 'orig_row_splits']
    '''

    orig_inputs = condition_input(orig_inputs, no_scaling = True)

    if pass_through:
        orig_inputs['orig_row_splits'] = orig_inputs['row_splits']
        orig_inputs['corr_rechit_energy'] = orig_inputs['rechit_energy']
        return orig_inputs

    coords = orig_inputs['prime_coords']
    x = orig_inputs['features']

    x = ValAndSign()(x)

    x = ScaledGooeyBatchNorm(
            trainable=trainable,
                viscosity=0.1,
                fluidity_decay=1e-1,#gets to 1 very quickly
                max_viscosity=1.)(x)

    orig_inputs['x_skip'] = x

    #prepare different embeddings for tracks and hits
    x_track = Dense(32, activation='elu', name=name+'emb_xtrack',trainable=trainable)(x)
    x_hit = Dense(32, activation='elu', name=name+'emb_xhit',trainable=trainable)(x)
    x = MixWhere()([orig_inputs['is_track'], x_track, x_hit]) #Concatenate()([coords,x,good_track_feat])

    good_track_feat = ScalarMultiply(1./20.)(SelectFeatures(9,10)(orig_inputs['features']))
    good_track_feat = Where(0.,'==0')([orig_inputs['is_track']  ,good_track_feat])
    good_track_feat = Abs()(good_track_feat)
    orig_inputs['track_dec_z'] = good_track_feat

    # basically a mini gravnet here

    coords = ElementScaling(name=name+'es1',trainable=trainable)(coords)
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_np_knn',
                    min_bins=20)([coords,orig_inputs['row_splits']])#hard code it here, this is optimised given our datasets
    x = DistanceWeightedMessagePassing([32],name=name+'np_dmp1',
                                       activation=activation,
                                       trainable=trainable)([x,nidx,dist])# hops are rather light
    x = Dense(32,activation=activation,name=name+'dense_np_1a',trainable=trainable)(x)
    x = Dense(32,activation=activation,name=name+'dense_np_1b',trainable=trainable)(x)

    #correction maxes out at 0-2
    orig_inputs['corr_rechit_energy'] = Multiply()([
        Dense(1,activation='tanh',name=name+'ecorr1',
              kernel_initializer='zeros',
              activity_regularizer=tf.keras.regularizers.L2(0.5),#regularize quite strongly
              )(x),
        orig_inputs['rechit_energy']])

    orig_inputs['corr_rechit_energy'] = Add(
        )([orig_inputs['corr_rechit_energy'] ,
                                               orig_inputs['rechit_energy']])

    orig_inputs['corr_rechit_energy'] = LLEnergySums(
        name=name+'LLEnergySums',
        scale = 50.,
        active=trainable,record_metrics=record_metrics
        )([
        orig_inputs['corr_rechit_energy'], orig_inputs['is_track'],
        orig_inputs['t_idx'],
        orig_inputs['t_energy'],
        orig_inputs['t_is_unique'],
        orig_inputs['t_pid'],
        orig_inputs['row_splits']
        ])


    no_noise = noise_filter_block(orig_inputs, x, name, trainable,
                                  record_metrics, noise_threshold,
                                  orig_inputs['row_splits'],
                                  print_time=print_time,
                                  debugplots_after=debugplots_after,
                                  debug_outdir=debug_outdir,
                                  publish = publish)

    coords = no_noise['prime_coords']

    rs = no_noise['row_splits']
    energy = no_noise['rechit_energy']
    is_track = no_noise['is_track']
    good_track_feat = no_noise['track_dec_z']
    t_energy = no_noise['t_energy']
    t_idx = no_noise['t_idx']
    t_spec_w = no_noise['t_spectator_weight']


    #don't expect too much here
    x = Concatenate()([no_noise['x_skip'],no_noise['x'],coords,good_track_feat])
    # allow different  embeddings again
    x_track = Dense(32,  activation='elu',name=name+'d_xtrack',trainable=trainable)(x)
    x_hit = Dense(32,  activation='elu',name=name+'d_xhit',trainable=trainable)(x)
    x_c = MixWhere()([is_track, x_track, x_hit]) #Concatenate()([coords,x,good_track_feat])

    #this is important
    x_c = Dense(32, activation='elu', name=name+'d_coord1',trainable=trainable)(x_c)
    coords_m = Dense(3, activation='tanh',name=name+'d_coord2mult',trainable=trainable)(x_c)
    coords_a = Dense(3, name=name+'d_coord2add',trainable=trainable)(x_c)
    coords = Add()([coords_a,coords])
    coords = Multiply()([coords_m,coords])

    coords = ElementScaling(name=name+'_pn_es1',trainable=trainable)(coords)

    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_knn',
                    min_bins=20)([coords,orig_inputs['row_splits']])#hard code it here, this is optimised given our datasets


    if debugplots_after>0:
        coords = PlotCoordinates(plot_every=debugplots_after,
                                    outdir=debug_outdir,name=name+'_pre_coords')(
                                        [coords,
                                         energy,
                                         is_track,
                                         nidx,
                                             t_idx,rs])

    coords = LLClusterCoordinates(active = trainable,
        name = name+'_LLClusterCoordinates',record_metrics = record_metrics,
        scale=1.
        )([coords,t_idx,t_spec_w,energy, rs])

    if fill_space_loss > 0:
        coords = LLFillSpace(scale=fill_space_loss,
                         active = trainable,
                         record_metrics = record_metrics,
                         name = name+'_LLFillSpace')([coords,rs])

    x = Concatenate()([no_noise['x'],no_noise['x_skip']])
    #sort by distance - only important later
    dist,nidx = SortAndSelectNeighbours(K=-1)([dist,nidx])
    scale = Dense(1,name=name+'dense1_scale',trainable=trainable)(x)
    dist = LocalDistanceScaling()([dist,scale])
    x_gn = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
                                       activation=activation,
                                       trainable=trainable)([x,nidx,dist])# hops are rather light

    #make this one a direct attention mechanism actually
    n_itheads = 6
    if True:
        for i in range(n_itheads):
            # sorting becomes relevant; the gradient for the coordinate space is explicit here,
            # so no need to have an implicit one through distance weighting
            # but needs position embeddings
            pos_emb = Dense(4,activation=activation,trainable=trainable, name=name+'pos_emb'+str(i))(x_gn)
            pos_emb = SelectFromIndicesWithPad()([nidx, pos_emb])
            pos_emb = Flatten()(pos_emb)
            dist = Concatenate()([x_gn, pos_emb])
            dist = Dense(K+1,activation='relu',name=name+'dense2_att'+str(i),trainable=trainable)(dist)
            x_gn = DistanceWeightedMessagePassing([32],name=name+'dmp_att'+str(i),
                                               activation=activation,
                                               trainable=trainable)([x_gn,nidx,dist])# hops are rather light
    else:
        #gives up on order invariance in neighbours
        x_gn = Dense(64, activation=activation, name=name+'_d_pre_att')(x_gn)
        x_gn = AttentionMP(
            4 * [16],
            K+1,
            4 #position encoding
            )([x_gn,nidx])
        #x_att_skip = Dense(64, activation=activation, name=name+'_d_pre_att')(x_gn)# 32 * 2 * 6 = 384
        #x_gn = LocalGravNetAttention(6, 4, 32, name=name+'_gnatt1')([x_gn, nidx, dist])
        #x_gn = Dense(64,activation=activation,name=name+'_d_att1',trainable=trainable)(x_gn)
        #x_gn = Add()([x_gn,x_att_skip])
        #x_gn = LocalGravNetAttention(6, 4, 32, name=name+'_gnatt2')([x_gn, nidx, dist])
        #x_gn = Dense(64,activation=activation,name=name+'_d_att2',trainable=trainable)(x_gn)
        #x_gn = Add()([x_gn,x_att_skip])

    x = Dense(32,activation=activation,name=name+'dense1a',trainable=trainable)(x_gn)
    x = Dense(32,activation=activation,name=name+'dense1b',trainable=trainable)(x)

    if flat_edges:
        x_e = Dense(24,activation=activation,name=name+'dense_x_e',trainable=trainable)(x)
        x_e = SelectFromIndicesWithPad()([nidx, x_e])#this will be big
        x_e = Flatten()(x_e)
        x_e = Dense(16*K,activation=activation,name=name+'dense_flat_x_e1',trainable=trainable)(x_e)
        x_e = Dense(16*K,activation=activation,name=name+'dense_flat_x_e2',trainable=trainable)(x_e)
        x_e = Reshape((K,16))(x_e)
    else:
        x_e = Concatenate()([x, no_noise['x_skip'] , x_gn])
        x_e = Dense(32,activation=activation,name=name+'dense_x_e',trainable=trainable)(x_e) #96
        x_e = EdgeCreator()([nidx, x_e])
        x_e = Dense(32,activation=activation,name=name+'dense_x_e1',trainable=trainable)(x_e) #64
        x_e = Dense(16,activation=activation,name=name+'dense_x_e2',trainable=trainable)(x_e) #32
        x_e = Dense(16,activation=activation,name=name+'dense_x_e3',trainable=trainable)(x_e) #32
        x_e_istrack = EdgeCreator()([nidx, is_track])
        x_e = Concatenate()([x_e, x_e_istrack]) #make it very easy

    x_e = Dense(1,activation='sigmoid',name=name+'_ed3',trainable=trainable)(x_e)

    #not just where so that it ca be switched off without truth
    cluster_tidx = MaskTracksAsNoise(active=trainable)([t_idx,is_track])

    x_e = LLEdgeClassifier( name = name+'_LLEdgeClassifier',active=trainable,record_metrics=record_metrics,
            scale=5.,#high scale
            print_batch_time=print_time,
            #fp_weight = 0.9, #0.9,
            #hifp_penalty = hifp_penalty,
            lin_e_weight=lin_edge_weight
            )([x_e,nidx,cluster_tidx, t_spec_w, t_energy, energy])

    if debugplots_after > 0:
        x_e = PlotEdgeDiscriminator(plot_every=debugplots_after,
                                    publish=publish,
                                        outdir=debug_outdir,name=name+'_edges')([x_e,nidx,cluster_tidx,no_noise['t_energy']])


    ##### no trainable weights beyond this point

    if pre_train_mode:
        return {'x_e':x_e, 'coords': coords}

    sel_nidx = EdgeSelector(
            threshold=reduction_threshold
            )([x_e,nidx])
    ### pre-train done


    den_offset = 10.#120.
    hierarchy = GroupScoreFromEdgeScores(den_offset = den_offset)([x_e,sel_nidx])

    g_sel_nidx, g_sel, group_backgather, g_sel_rs = NeighbourGroups(threshold = 1e-3,
        return_backscatter=False,
        #print_reduction=True,
        record_metrics = False)([hierarchy, sel_nidx, rs])

    #only after groups have been built
    sel_t_spec_w, sel_t_idx, unamb_score = AmbiguousTruthToNoiseSpectator(
        return_score = True,
        record_metrics=record_metrics
        )([sel_nidx, t_spec_w, t_idx, energy])

    g_sel = MLReductionMetrics(
        name=name+'_reduction_0',
        record_metrics = record_metrics
        )([g_sel,t_idx,no_noise['t_energy'],no_noise['row_splits'],g_sel_rs])


    g_sel_sel_nidx = SelectFromIndices()([g_sel,g_sel_nidx])

    #create selected output truth
    out={}
    for k in no_noise.keys():
        if 't_' == k[0:2]:
            out[k] = SelectFromIndices()([g_sel,no_noise[k]])

    #consider ambiguities
    out['t_idx'] = SelectFromIndices()([g_sel,sel_t_idx])
    out['t_spectator_weight'] = SelectFromIndices()([g_sel,sel_t_spec_w])
    out['t_unamb_score'] = SelectFromIndices()([g_sel,unamb_score])

    ## create reduced features

    x_to_flat = no_noise['x_skip']#full input info

    #this has output dimension now
    x_flat_o = SelectFromIndicesWithPad()([g_sel_sel_nidx, x_to_flat])
    x_flat_o = Flatten()(x_flat_o)
    x_o = x_flat_o

    #explicitly sum energy
    energy_o = AccumulateNeighbours('sum')([energy, g_sel_nidx])
    energy_o = SelectFromIndices()([g_sel,energy_o])

    corr_rechit_energy = AccumulateNeighbours('sum')([no_noise['corr_rechit_energy'], g_sel_nidx])
    corr_rechit_energy = SelectFromIndices()([g_sel,corr_rechit_energy])

    #build preselection coordinates
    coords_o = AccumulateNeighbours('mean')([coords, g_sel_nidx, energy])
    coords_o = SelectFromIndices()([g_sel,coords_o])

    prime_coords_o = AccumulateNeighbours('mean')([no_noise['prime_coords'], g_sel_nidx, energy])
    prime_coords_o = SelectFromIndices()([g_sel,prime_coords_o])

    #pass original features
    mean_orig_feat = AccumulateNeighbours('mean')([no_noise['orig_features'], g_sel_nidx, energy])
    mean_orig_feat = SelectFromIndices()([g_sel,mean_orig_feat])

    ## selection done work on selected ones

    #was not clustered
    is_track_o = SelectFromIndices()([g_sel,is_track])


    #add to dict:
    out['coords'] = coords_o
    out['prime_coords'] = prime_coords_o
    out['rechit_energy'] = energy_o
    out['corr_rechit_energy'] = corr_rechit_energy
    out['features'] = x_o
    out['orig_features'] = mean_orig_feat
    out['is_track'] = is_track_o

    #trained to be 100% efficient for tracks
    out['track_dec_z'] = SelectFromIndices()([g_sel,no_noise['track_dec_z']])

    out['scatterids'] = [no_noise['scatterids'], group_backgather]

    out['orig_row_splits'] = orig_inputs['row_splits']
    out['row_splits'] = g_sel_rs
    out['no_noise_row_splits'] = no_noise['row_splits']

    if debugplots_after>0:
        out['coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_after_red')(
                                            [out['coords'],
                                             out['rechit_energy'],
                                             out['t_idx'],g_sel_rs])


    return out



def tiny_intermediate_condensation(
        x, rs, energy, t_specweight, t_idx,
        trainable=True,
        record_metrics=True,
        cluster_dims=3,
        name='intermediate_condensation'
        ):

    beta = Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_beta')(x)
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(x))
    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(x)

    beta = LLBasicObjectCondensation(
        q_min=.2,#high qmin here
        implementation = 'std',
        record_batch_time = record_metrics,
        scale = 0.2,
        active=trainable,
        print_batch_time = True,
        record_metrics = record_metrics,
        use_average_cc_pos=0.1 #small
        )([beta, ccoords, d,t_specweight,t_idx, rs])

    ch_idx, c_idx, rev_r, revflat,\
     asso_idx = RaggedCreateCondensatesIdxs(t_d = 0.25, t_b = 0.1,
                                                     return_thresholds=False)([beta,ccoords,d,rs])

    c_x = RaggedSelectFromIndices()([x, c_idx])

    return RaggedSelectFromIndices()([c_x, rev_r])


def intermediate_condensation(
        x, rs, energy, t_specweight, t_idx,
        trainable=True,
        record_metrics=True,
        cluster_dims=3,
        name='intermediate_condensation'
        ):

    beta = Dense(1,activation='sigmoid',trainable=trainable,name=name+'_proto_beta')(x)
    d = ScalarMultiply(2.)(Dense(1, activation='sigmoid',name=name+'dense_d')(x))
    ccoords = Dense(cluster_dims,name=name+'dense_ccoords',trainable=trainable)(x)

    beta = LLBasicObjectCondensation(
        q_min=1.,#high qmin here
        implementation = 'std',
        record_batch_time = record_metrics,
        #scale = 0.1,
        active=trainable,
        print_batch_time = False,
        print_loss=True,
        record_metrics = record_metrics,
        use_average_cc_pos=0.5 #small
        )([beta, ccoords, d,t_specweight,t_idx, rs])


    #low thresholds
    ch_idx, c_idx, _, revflat,\
     asso_idx = RaggedCreateCondensatesIdxs(t_d = 0.25, t_b = 0.1,
                                            collapse_noise=False,
                                                     return_thresholds=False)([beta,ccoords,d,rs])
    # t_d, t_b
    #if False:
    #    beta = LLOCThresholds(
    #               name=name+'_ll_oc_thresholds',
    #               active=trainable,
    #               highest_t_d = 1.,
    #               record_batch_time = record_metrics,
    #               lowest_t_b = 0.5,
    #               purity_target = 0.8,
    #               record_metrics=record_metrics)([beta, asso_idx, d,
    #                                               ccoords , t_idx, t_d, t_b])
    #
    #
    #else:
    #    beta = LLFullOCThresholds(
    #        name=name+'_ll_full_oc_thresholds',
    #        active=trainable,
    #        purity_weight = 0.9,
    #         )([beta, ccoords, d, c_idx, ch_idx,
    #            energy, t_idx, t_depe, rs, t_d, t_b])

    x = Concatenate()([x, StopGradient()(beta)])
    x = Dense(64, activation='elu', name=name+'_predense')(x)
    c_x = RaggedSelectFromIndices()([x, c_idx])
    ch_x = RaggedSelectFromIndices()([x, ch_idx])

    #print('c_x, ch_x',c_x.shape, ch_x.shape)
    c_x_skip_flat,_ = RaggedToFlatRS()(c_x)
    for c in ['mean','max']:
        ch_x = RaggedMixHitAndCondInfo('add')([ch_x, c_x])
        ch_x = RaggedDense(64, activation='elu', name=name+'_rdense1_'+c)(ch_x)
        c_x = RaggedCollapseHitInfo(c)(ch_x)

    #exchange through condensation points
    xf, xfrs = RaggedToFlatRS()(c_x)
    xf = Concatenate()([xf, c_x_skip_flat])

    if True:
        xf, _, gnnidx, gndist = RaggedGravNet(n_neighbours=32,
                                                     n_dimensions=3,
                                                     n_filters=64,
                                                     n_propagate=64,
                                                     record_metrics=True,
                                                     feature_activation='elu',
                                                     name=name+'_cpgn1',
                                                     debug=True
                                                     )([xf, xfrs])

        #xf = DistanceWeightedMessagePassing([32,32],name=name+'dmp1',
        #                                   trainable=trainable,
        #                                   activation='elu')([xf,gnnidx,gndist])

    xf = Dense(64, activation='elu',name=name+'_d1',trainable=trainable)(xf)
    # don't make it ragged again. no reason
    #c_x = FlatRSToRagged()([xf, xfrs])
    #
    #for c in ['mean','max']:
    #    ch_x = RaggedMixHitAndCondInfo('add')([ch_x, c_x])
    #    ch_x = RaggedDense(64, activation='elu', name=name+'_rdense2_'+c)(ch_x)
    #    c_x = RaggedCollapseHitInfo(c)(ch_x)
    #
    #c_x,_ = RaggedToFlatRS()(c_x)
    #print('c_x',c_x.shape)
    xf = Concatenate()([xf, c_x_skip_flat])
    #backgather to all hits, x is flat again
    x = RaggedSelectFromIndices()([xf, revflat])
    print('x',x.shape)
    #print('x out',x.shape, 'revflat',revflat.shape)
    return x #RaggedToFlatRS()(c_x)



def pre_graph_condensation(
        orig_inputs,
        debug_outdir='',
        trainable=False,
        name='pre_graph_condensation',
        debugplots_after=-1,
        record_metrics=True,
        K_loss = 96,
        publish=None,
        dynamic_spectators=True,
        first_call=True):

    activation = 'elu'
    K = 16

    #orig_inputs = condition_input(orig_inputs, no_scaling = True)


    #just check while in training mode
    if trainable:
        for k in orig_inputs.keys():
            if (not 't_' == k[0:2]) and (not 'row_splits' in k):
                orig_inputs[k] = CheckNaN(name=name+'_pre_check_'+k)(orig_inputs[k])

    x = orig_inputs['features'] # coords
    coords = orig_inputs['prime_coords']
    prime_coords = coords
    rs = orig_inputs['row_splits']
    energy = orig_inputs['rechit_energy']
    is_track = orig_inputs['is_track']


    x = ScaledGooeyBatchNorm2(
            name=name+'_cond_batchnorm_a',
            record_metrics = record_metrics)([x, is_track])

    x = ScaledGooeyBatchNorm2(
            name=name+'_cond_batchnorm_b',
            invert_condition = True,
            record_metrics = record_metrics)([x, is_track])

    x_in = x

    if trainable:
        for k in orig_inputs.keys():
            if (not 't_' == k[0:2]) and (not 'row_splits' in k):
                orig_inputs[k] = CheckNaN(name=name+'_pre_check_postnorm_'+k)(orig_inputs[k])

    if first_call:

        #pre-processing
        x_track = Dense(32, activation='elu', name=name+'emb_xtrack',trainable=trainable)(x)
        #x_track = Dense(32, activation='elu', name=name+'emb_xtrack1',trainable=trainable)(x_track)
        x_hit = Dense(32, activation='elu', name=name+'emb_xhit',trainable=trainable)(x)
        #x_hit = Dense(32, activation='elu', name=name+'emb_xhit1',trainable=trainable)(x_hit)
        x = MixWhere()([is_track, x_track, x_hit]) #Concatenate()([coords,x,good_track_feat])
        x = ScaledGooeyBatchNorm2(name = name+'_batchnorm_0a', trainable=trainable,
                                  record_metrics = record_metrics)(x)

    x_skip = x

    #gravnet block # 6 dims orig
    coords = expand_coords_if_needed(coords, x, 4, name=name+'_exp_coords', trainable=trainable)

    #simple gravnet
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_np_knn',
                    min_bins=20)([coords,orig_inputs['row_splits']])#hard code it here, this is optimised given our datasets

    x = Concatenate()([SelfAttention(name = name+'_selfatt1')(x),x])
    x = DistanceWeightedMessagePassing([32,32],name=name+'np_dmp1',
                                        activation='elu',#keep output in check
                                        trainable=trainable)([x,nidx,dist])# hops are rather light

    x = ScaledGooeyBatchNorm2(name = name+'_batchnorm1', trainable=trainable, record_metrics = record_metrics)(x)

    x = Concatenate()([SelfAttention(name = name+'_selfatt2')(x),x])
    x = Dense(32,activation=activation,name=name+'dense_np_0b',trainable=trainable)(x)

    x = DistanceWeightedMessagePassing([32,32],name=name+'np_dmp2',
                                        activation='elu',#keep output in check
                                        trainable=trainable)([x,nidx,dist])# hops are rather light

    x = ScaledGooeyBatchNorm2(name = name+'_batchnorm2', trainable=trainable, record_metrics = record_metrics)(x)

    x = Concatenate()([SelfAttention(name = name+'_selfatt3')(x),x])
    x = Dense(32,activation=activation,name=name+'dense_np_0c',trainable=trainable)(x)


    x = DistanceWeightedMessagePassing([32,32],name=name+'np_dmp3',
                                        activation='elu',#keep output in check
                                        trainable=trainable)([x,nidx,dist])# hops are rather light

    #x = Dense(46,activation=activation,name=name+'dense_np_1b',trainable=trainable)(x)
    x = Dense(32,activation='elu',name=name+'dense_np_2a',trainable=trainable)(x)
    x = RaggedGlobalExchange(skip_min=True)([x,rs]) #not a lot of information in minimum due to elu activation
    x = ScaledGooeyBatchNorm2(name = name+'_batchnorm3', trainable=trainable, record_metrics = record_metrics)(x)
    #make sure things don't explode
    x = Dense(32,activation='elu',name=name+'dense_np_2b',trainable=trainable)(x)
    x = Concatenate()([x, prime_coords])
    x = ScaledGooeyBatchNorm2(name = name+'_batchnorm4',
                              trainable=trainable, record_metrics = record_metrics)(x)


    if dynamic_spectators:
        tmp_spec = Dense(1, activation = 'sigmoid', name = name+'_dyn_specw')(x)
        orig_inputs['t_spectator_weight'] = LLValuePenalty(active = trainable,
                                                    record_metrics=record_metrics)(tmp_spec)

    #now go for it ############### main condensation part below

    score = Dense(1, activation='sigmoid',name=name+'_gc_score', trainable=trainable)(x)
    coords = Dense(3, name=name+'_xyz_cond', use_bias = False,
                   #kernel_initializer = 'zeros',
                   trainable=trainable)(x)

    coords = LLClusterCoordinates(
                downsample=10000,
                record_metrics = record_metrics,
                active=trainable,
                scale = 1.,
                ignore_noise = True, #this is filtered by the graph condensation anyway
                print_batch_time=True,
                hinge_mode = True
                )([coords, orig_inputs['t_idx'], orig_inputs['t_spectator_weight'],
                                            score, rs ])

    if debugplots_after > 0:
        coords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_cond_coords',
                                        publish=publish)(
                                            [coords,
                                             Where(0.1)([is_track,energy]),
                                             orig_inputs['t_idx'],orig_inputs['row_splits']])

    score = LLGraphCondensationScore(
        record_metrics = record_metrics,
        K=K_loss,
                active=trainable,
            penalty_fraction=0.5,
            low_energy_cut = 1.,
            print_loss = trainable
            )([score, coords, orig_inputs['t_idx'], orig_inputs['t_energy'], rs])

    trans_a = CreateGraphCondensation(
            score_threshold = 0.5,
            K=5
            )(score,coords,rs,
              always_promote = is_track)


    trans_a = MLGraphCondensationMetrics(
        name = name + '_graphcondensation_metrics',
        record_metrics = record_metrics,
        )(trans_a, orig_inputs['t_idx'], orig_inputs['t_energy'])

    #these also act as a fractional noise filter
    x_e = CreateGraphCondensationEdges(
                 edge_dense=[32,16,16],#[64,32,32],
                 pre_nodes=8,#12,
                 K=5,
                                       trainable=trainable,
                                       name=name+'_gc_edges')(x, trans_a)

    x_e = LLGraphCondensationEdges(
        active=trainable,
        record_metrics=record_metrics
        )(x_e, trans_a, orig_inputs['t_idx'])

    trans_a = InsertEdgesIntoTransition()(x_e, trans_a)


    out = {}

    out['prime_coords'] = PushUp(add_self=True)(orig_inputs['prime_coords'], trans_a, weight = energy)
    out['coords'] = PushUp(add_self=True)(orig_inputs['coords'], trans_a, weight = energy)
    out['rechit_energy'] = PushUp(mode='sum', add_self=True)(energy, trans_a)

    # for plotting and understanding
    if debugplots_after > 0:

        coords = PlotGraphCondensation(
                     plot_every = debugplots_after,
                     outdir= debug_outdir ,
                     publish = publish
                     )([coords,
                        Where(0.1)([is_track,energy]),#make tracks small in plot
                        trans_a['weights_down'], trans_a['nidx_down'], rs])

        orig_inputs['prime_coords'] = PlotGraphCondensation(
                     plot_every = debugplots_after,
                     outdir= debug_outdir ,
                     publish = publish
                     )([orig_inputs['prime_coords'],
                        Where(0.1)([is_track,energy]),#make tracks small in plot
                        trans_a['weights_down'], trans_a['nidx_down'], rs])

        orig_inputs['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = debugplots_after//2,
                     outdir= debug_outdir ,
                     name = name + '_efficiency',
                     publish = publish)(orig_inputs['t_energy'], orig_inputs['t_idx'], trans_a)


        out['prime_coords'] = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_after_coords',
                                        publish=publish)(
                                            [out['prime_coords'],
                                             out['rechit_energy'],
                                             SelectUp()(orig_inputs['t_idx'], trans_a),
                                             trans_a['rs_up']])


    out['down_features'] = Concatenate()([x,x_skip,x_in]) # this is for further "bouncing"
    out['is_track'] = SelectUp()(is_track, trans_a)
    out['cond_coords'] = coords
    out['row_splits'] = trans_a['rs_up']

    for k in orig_inputs.keys():
        if 't_' == k[0:2]:
            out[k] = SelectUp()(orig_inputs[k],trans_a)


    print('pre condensation outputs:', out.keys())

    #just check while in training mode
    if trainable:
        for k in out.keys():
            if (not 't_' == k[0:2]) and (not k == 'row_splits'):
                out[k] = CheckNaN(name=name+'_post_check_'+k)(out[k])

    return out, trans_a


def mini_pre_graph_condensation(
        orig_inputs,
        debug_outdir='',
        trainable=False,
        name='pre_graph_condensation',
        debugplots_after=-1,
        record_metrics=True,
        produce_output = True,
        K_loss = 48,
        score_threshold = 0.8,
        low_energy_cut = 2.,
        publish=None,
        dynamic_spectators=True,
        coords = None,
        first_call=True):

    activation = 'elu'
    K = 16

    #orig_inputs = condition_input(orig_inputs, no_scaling = True)


    #just check while in training mode
    if trainable:
        for k in orig_inputs.keys():
            if (not 't_' == k[0:2]) and (not 'row_splits' in k):
                orig_inputs[k] = CheckNaN(name=name+'_pre_check_'+k)(orig_inputs[k])

    x = orig_inputs['features'] # coords
    if coords is None:
        coords = orig_inputs['prime_coords']
    rs = orig_inputs['row_splits']
    energy = orig_inputs['rechit_energy']
    is_track = orig_inputs['is_track']

    x = ConditionalScaledGooeyBatchNorm(
            name=name+'_cond_batchnorm',
            record_metrics = record_metrics)([x, is_track])

    x_in = x

    if trainable:
        for k in orig_inputs.keys():
            if (not 't_' == k[0:2]) and (not 'row_splits' in k):
                orig_inputs[k] = CheckNaN(name=name+'_pre_check_postnorm_'+k)(orig_inputs[k])

    if first_call:

        #pre-processing
        x_track = Dense(16, activation='elu', name=name+'emb_xtrack',trainable=trainable)(x)
        #x_track = Dense(32, activation='elu', name=name+'emb_xtrack1',trainable=trainable)(x_track)
        x_hit = Dense(16, activation='elu', name=name+'emb_xhit',trainable=trainable)(x)
        #x_hit = Dense(32, activation='elu', name=name+'emb_xhit1',trainable=trainable)(x_hit)
        x = MixWhere()([is_track, x_track, x_hit]) #Concatenate()([coords,x,good_track_feat])
        x = ScaledGooeyBatchNorm2(name = name+'_batchnorm_0a', trainable=trainable,
                                  record_metrics = record_metrics)(x)

    x_skip = x

    #simple scaling
    coords = ElementScaling(name=name+'_es_coords', trainable=trainable)(coords)
    #simple gravnet
    nidx,dist = KNN(K=K,record_metrics=record_metrics,name=name+'_np_knn',
                    min_bins=20)([coords,orig_inputs['row_splits']])#hard code it here, this is optimised given our datasets

    x = DistanceWeightedMessagePassing([8,8,16,16,32,32],name=name+'np_dmp1',
                                        activation='elu',#keep output in check
                                        trainable=trainable)([x,nidx,dist])# hops are rather light

    x = ScaledGooeyBatchNorm2(name = name+'_batchnorm3', trainable=trainable,
                              record_metrics = record_metrics)(x)
    #make sure things don't explode
    x = Dense(48,activation='elu',name=name+'dense_np_2b',trainable=trainable)(x)
    x = ScaledGooeyBatchNorm2(name = name+'_batchnorm4',
                              trainable=trainable, record_metrics = record_metrics)(x)

    if dynamic_spectators:
        tmp_spec = Dense(1, activation = 'sigmoid', name = name+'_dyn_specw')(x)
        orig_inputs['t_spectator_weight'] = LLValuePenalty(active = trainable,
                                                    record_metrics=record_metrics)(tmp_spec)

    #now go for it ############### main condensation part below

    score = Dense(1, activation='sigmoid',name=name+'_gc_score', trainable=trainable)(x)
    coords = Add()([coords, Dense(3, name=name+'_xyz_cond', use_bias = False,
                   #kernel_initializer = 'zeros',
                   trainable=trainable)(x) ] )

    coords = LLClusterCoordinates(
                downsample=5000,
                record_metrics = record_metrics,
                active=trainable,
                scale = 1.,
                ignore_noise = True, #this is filtered by the graph condensation anyway
                print_batch_time=True,
                hinge_mode = True
                )([coords, orig_inputs['t_idx'], orig_inputs['t_spectator_weight'],
                                            score, rs ])

    if debugplots_after > 0:
        coords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_cond_coords',
                                        publish=publish)(
                                            [coords,
                                             Where(0.1)([is_track,energy]),
                                             orig_inputs['t_idx'],orig_inputs['row_splits']])

    score = LLGraphCondensationScore(
        record_metrics = record_metrics,
        K=K_loss,
                active=trainable,
            penalty_fraction=0.5,
            low_energy_cut = low_energy_cut,
            print_loss = trainable
            )([score, coords, orig_inputs['t_idx'], orig_inputs['t_energy'], rs])

    trans_a = CreateGraphCondensation(
            score_threshold = score_threshold,
            K=5
            )(score,coords,rs,
              always_promote = is_track)


    trans_a = MLGraphCondensationMetrics(
        name = name + '_graphcondensation_metrics',
        record_metrics = record_metrics,
        )(trans_a, orig_inputs['t_idx'], orig_inputs['t_energy'])

    #these also act as a fractional noise filter
    x_e = CreateGraphCondensationEdges2(name=name+'_gc_edges')(x, trans_a)

    x_e = LLGraphCondensationEdges(
        active=trainable,
        record_metrics=record_metrics
        )(x_e, trans_a, orig_inputs['t_idx'])

    trans_a = InsertEdgesIntoTransition()(x_e, trans_a)


    out = {}

    if produce_output:

        out['prime_coords'] = PushUp(add_self=True)(orig_inputs['prime_coords'], trans_a, weight = energy)
        out['coords'] = PushUp(add_self=True)(orig_inputs['coords'], trans_a, weight = energy)
        out['rechit_energy'] = PushUp(mode='sum', add_self=True)(energy, trans_a)

        out['down_features'] = Concatenate()([x,x_skip,x_in]) # this is for further "bouncing"
        out['is_track'] = SelectUp()(is_track, trans_a)
        out['cond_coords'] = coords
        out['row_splits'] = trans_a['rs_up']

    # for plotting and understanding
    if debugplots_after > 0:

        orig_inputs['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = debugplots_after//2,
                     outdir= debug_outdir ,
                     name = name + '_efficiency',
                     publish = publish)(orig_inputs['t_energy'], orig_inputs['t_idx'], trans_a)


    for k in orig_inputs.keys():
        if 't_' == k[0:2]:
            out[k] = SelectUp()(orig_inputs[k],trans_a)


    print('pre condensation outputs:', out.keys())

    #just check while in training mode
    if trainable:
        for k in out.keys():
            if (not 't_' == k[0:2]) and (not k == 'row_splits'):
                out[k] = CheckNaN(name=name+'_post_check_'+k)(out[k])

    return out, trans_a

'''

Can define a 'cleaning' step:
- use LLGraphCondOCLoss to define beta as the information content per hit
- use beta as condensation score (threshold = ?)
- determine the edges as object association
- Push up, and thereby remove ambiguous hits
- threshold can even be learnt/dynamically adjusted using efficiency metrics

'''
def intermediate_graph_cleaning(
        orig_inputs, # x, coords, energy, is_track, score, t_idx, t_spectator_weight, t_energy, rs
        edge_dense = [64,64,64],
        edge_pre_nodes = 32,
        K = 5,
        score_threshold = 0.5,
        trainable=False,
        record_metrics = False,
        name = 'graph_cleaning'
        ):

    x = orig_inputs['x']
    score = orig_inputs['score']
    coords = orig_inputs['coords']
    is_track = orig_inputs['is_track']
    energy = orig_inputs['rechit_energy']
    rs = orig_inputs['row_splits']

    trans_a = CreateGraphCondensation(
            score_threshold = score_threshold,
            K=K
            )(score,coords,rs,
              always_promote = is_track)

    sum_energy = None
    if len(edge_dense):

        x_e = CreateGraphCondensationEdges(
                     edge_dense=edge_dense,
                     pre_nodes=edge_pre_nodes,
                     K=K, trainable=trainable,
                     name=name+'_gc_edges')(x, trans_a)

        x_e = LLGraphCondensationEdges(
            active=trainable,
            record_metrics=record_metrics
            )(x_e, trans_a, orig_inputs['t_idx'])

        trans_a = InsertEdgesIntoTransition()(x_e, trans_a)

        sum_energy = PushUp(mode='sum', add_self=True)(energy, trans_a)

    return trans_a, sum_energy


def intermediate_graph_condensation(
        orig_inputs, #features, is_track, rechit_energy, row_splits, t_idx, t_spectator_weight, t_energy
        n_cond_coords :int = 3,

        edge_dense = [64,64,64],
        edge_pre_nodes = 32,
        K = 5,
        K_loss = 32,
        score_threshold=0.5,


        debug_outdir='',
        trainable=False,
        name='int_gc',
        debugplots_after=-1,
        record_metrics=True,
        publish=None,
        dynamic_spectators=False):

    x = orig_inputs['features']
    rs = orig_inputs['row_splits']
    is_track = orig_inputs['is_track']
    energy = orig_inputs['rechit_energy']

    score = Dense(1, activation='sigmoid',name=name+'_gc_score', trainable=trainable)(x)
    coords = Dense(n_cond_coords, name=name+'_cond_coords', use_bias = False, trainable=trainable)(x)

    score = LLGraphCondensationScore(
        record_metrics = record_metrics,
        K=K_loss,
                active=trainable,
            penalty_fraction=0.5,
            low_energy_cut = 1. #allow everything below 1 GeV to be removed
            )([score, coords, orig_inputs['t_idx'], orig_inputs['t_energy'], rs])

    coords = LLClusterCoordinates(
                record_metrics = record_metrics,
                active=trainable,
                scale = 1.,
                ignore_noise = True, #this is filtered by the graph condensation anyway
                print_batch_time=False
                )([coords, orig_inputs['t_idx'], orig_inputs['t_spectator_weight'],
                                            score, rs ])

    trans_a = CreateGraphCondensation(
            score_threshold = score_threshold,
            K=K
            )(score,coords,rs,
              always_promote = is_track)

    trans_a = MLGraphCondensationMetrics(
        name = name + '_graphcondensation_metrics',
        record_metrics = record_metrics,
        )(trans_a, orig_inputs['t_idx'], orig_inputs['t_energy'])

    sum_energy = None

    if len(edge_dense):

        x_e = CreateGraphCondensationEdges(
                     edge_dense=edge_dense,
                     pre_nodes=edge_pre_nodes,
                     K=K, trainable=trainable,
                     name=name+'_gc_edges')(x, trans_a)

        x_e = LLGraphCondensationEdges(
            active=trainable,
            record_metrics=record_metrics
            )(x_e, trans_a, orig_inputs['t_idx'])

        trans_a = InsertEdgesIntoTransition()(x_e, trans_a)

        sum_energy = PushUp(mode='sum', add_self=True)(energy, trans_a)


    if debugplots_after > 0:

        orig_inputs['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = debugplots_after,
                     outdir= debug_outdir ,
                     publish = publish)(orig_inputs['t_energy'], orig_inputs['t_idx'], trans_a)

    #select truth
    out_truth={}
    for k in orig_inputs.keys():
        if 't_' == k[0:2]:
            out_truth[k] = SelectUp()(orig_inputs[k],trans_a)

    return trans_a, out_truth, sum_energy


def mini_tree_create(
        score,
        coords,
        rs,

        t_idx, t_energy, 

        is_track = None,

        K = 5,

        K_loss = 64,
        score_threshold=0.5,
        low_energy_cut = 4.,  #allow everything below 4 GeV to be removed
        
        record_metrics = False,
        trainable = False,
        name = 'tree_creation_0',
        always_record_reduction = True,

        cleaning_mode = False #changes the score loss to the cleaning loss
        ):
    '''
    provides the loss needed and the condensation graph
    Does not contain any learnable parameters whatsoever.
    trainable only turns on the losses
    '''

    if cleaning_mode:
        lcc_input = [coords, t_idx,  score, score, rs ]
    else:
        lcc_input = [coords, t_idx,  score, rs ]

    # right now this would scale with score (makes sense); could it be useful to cut that gradient?
    coords = LLClusterCoordinates(
                record_metrics = record_metrics,
                active=trainable,
                scale = .1, #scale this down, this will not be the main focus
                ignore_noise = True, #this is filtered by the graph condensation anyway
                print_batch_time=False,
                specweight_to_weight = True,
                downsample=30000, #downsample to 30k
                )(lcc_input) #score is not affected here

    if cleaning_mode:
        score = LLValuePenalty(1., active = trainable)(score)
    else:
        if is_track is not None:
            orig_score = score
            score = Where()([is_track, ZerosLike()(score),score]) #this will prevent collapse to a single track per particle

        score = LLGraphCondensationScore(
            record_metrics = record_metrics,
            K=K_loss,
                    active=trainable,
                penalty_fraction=0.5,#doesn't matter in current implementation
                low_energy_cut = low_energy_cut,
                name = name + '_score'
                )([score, coords, t_idx, t_energy, rs])
        
        if is_track is not None:
            score = Where()([is_track, orig_score, score]) #technically doesn't matter as tracks are always promoted

    trans_a = CreateGraphCondensation(
            score_threshold = score_threshold,
            K=K,
            name=name+'_tree'
            )(score,coords,rs, always_promote = is_track)

    trans_a = MLGraphCondensationMetrics(
        name = name + '_metrics',
        record_metrics = record_metrics or always_record_reduction
        )(trans_a, t_idx, t_energy)
    
    return trans_a


def mini_tree_clustering(
        pre_inputs : dict, # features, rechit_energy, row_splits, t_idx, 

        trans_a : GraphCondensation,

        edge_dense = [64,64,64],
        edge_pre_nodes = 32,

        record_metrics = False,
        trainable = False,
        name = 'tree_clustering_0',
        produce_output = True # turn off for fast pretraining
        ):
    
    energy = pre_inputs['rechit_energy']

    x = Dense(edge_pre_nodes, activation='elu', 
              kernel_initializer='he_normal',
              name=name+'_enc', trainable = trainable)(pre_inputs['features'])
    x_e = AddNeighbourDiff()(x, trans_a)

    for i, nodes in enumerate(edge_dense):
        x_e = Dense(nodes, activation='elu', 
              kernel_initializer='he_normal',
              name=name+f'_edge_dense_{i}', trainable = trainable)(x_e)
    #break them down to K+1
    x_e = Dense(trans_a.K()+1, activation='sigmoid', 
              name=name+'_edge_dense_final', trainable = trainable)(x_e)

    x_e = LLGraphCondensationEdges(
            active=trainable,
            record_metrics=record_metrics
            )(x_e, trans_a, pre_inputs['t_idx'])

    x_e = StopGradient()(x_e) #edges are purely learned from explicit loss
    trans_a = InsertEdgesIntoTransition()(x_e, trans_a) #this normalises them to sum=1-noise fraction

    out={}

    if produce_output:
        # now push up, some explicit:
        # explicit_keys = ['prime_coords', 'coords', 'rechit_energy', 'features', 'is_track', 'row_splits']

        out['prime_coords'] = PushUp(add_self=True)(pre_inputs['prime_coords'], trans_a, weight = energy)
        out['coords'] = PushUp(add_self=True)(pre_inputs['coords'], trans_a, weight = energy)
        out['rechit_energy'] = PushUp(mode='sum', add_self=True)(energy, trans_a)
        ew_feat = PushUp(add_self=False)(pre_inputs['features'],trans_a, weight = energy)
        w_feat = PushUp(add_self=False)(pre_inputs['features'],trans_a)
        x_sel = SelectUp()(pre_inputs['features'], trans_a)
        out['features'] = Concatenate()([x_sel, ew_feat, w_feat])
    
        out['is_track'] = SelectUp()(pre_inputs['is_track'], trans_a)
        out['row_splits'] = trans_a['rs_up']

        # now explicit pass through
        out['orig_features'] = pre_inputs['orig_features']
        out['orig_row_splits'] = pre_inputs['orig_row_splits']

        for k in pre_inputs.keys(): # pass through truth
            if 't_' == k[0:2]:
                out[k] = SelectUp()(pre_inputs[k],trans_a)

        return out, trans_a
    
    #this is a dummy return
    return {}, trans_a

def GravNet_plus_TEQMP(name,
                       x, cprime, hit_energy, t_idx,
                       rs, 
                       d_shape, 
                       n_neighbours,
                       debug_outdir, 
                       plot_debug_every, 
                       debug_publish = None,
                       space_reg_strength=-1.,
                       n_gn_coords = 4,
                       teq_nodes = [64, 32, 16, 8],
                       return_coords = False,
                       trainable = True,
                       add_scaling = True
                       ):
    
    xgn, gncoords, gnnidx, gndist = RaggedGravNet(
                name = "GravNet_"+name, # 76929, 42625, 42625
            n_neighbours=n_neighbours,
            n_dimensions=n_gn_coords,
            n_filters=d_shape,
            n_propagate=d_shape,
            coord_initialiser_noise=1e-3,
            feature_activation='elu',
            trainable = trainable,
            
            )([x, rs])

    if space_reg_strength > 0:
        gndist = LLRegulariseGravNetSpace(name=f'gravnet_coords_reg_{name}' , 
                                          record_metrics=True,
                                          scale=space_reg_strength)([gndist, cprime, gnnidx])
    
    gncoords = PlotCoordinates(
            plot_every= plot_debug_every,
            outdir = debug_outdir,
            name=f'gncoords_{name}',
            publish = debug_publish
            )([gncoords, hit_energy, t_idx, rs])
    
    x = DummyLayer()([x, gncoords]) #just so the branch is not optimised away, anyway used further down
    x = Concatenate()([xgn, x])

    dscale = Dense(1, name=name+'_dscale', trainable = trainable)(x) #FIXME sigmoid needs to go
    #dscale = ScalarMultiply(2.)(dscale)
    #dscale = Multiply()([dscale, dscale]) # as distances are also quadratic
    gndist = LocalDistanceScaling()([gndist, dscale])
    
    x = TranslationInvariantMP(teq_nodes, 
                 layer_norm = True,
                 activation = None, #layer norm takes care
                 sum_weight = False,
                 name = name+'_teqmp',
                 trainable = trainable)([x, gnnidx, gndist])

    if return_coords:
        return Concatenate()([xgn, x]), gncoords
    return Concatenate()([xgn, x])

def tree_condensation_block(pre_processed,
                             debug_outdir='', plot_debug_every=-1, debug_publish = None,
                             name = 'tree_condensation_block',
                             trainable = False,
                             record_metrics = False,
                             produce_output = True,
                             always_record_reduction = True,
                             decouple_coords = False,

                             low_energy_cut = 4.,
                             
                             enc_nodes = 32,
                             gn_nodes = 16,
                             gn_neighbours = 16,
                             teq_nodes = [16,16],
                             
                             edge_dense = [32,16],
                             edge_pre_nodes = 16 ):

    prime_coords = pre_processed['prime_coords']
    is_track = pre_processed['is_track']
    rs = pre_processed['row_splits']
    energy = pre_processed['rechit_energy']
    t_idx = pre_processed['t_idx']
    x = pre_processed['features']
    
     #keeping this in check is useful, therefore tanh is actually a good choice
    x = Dense(enc_nodes, activation='tanh', name=name+'_enc', trainable = trainable)(x)
    x = Concatenate()([prime_coords,x])

    xgn, gn_coords = GravNet_plus_TEQMP(name + '_net', x, prime_coords, energy, t_idx, rs, 
                                   gn_nodes, #nodes
                                   gn_neighbours, #neighbours
                                   debug_outdir, plot_debug_every,
                                   debug_publish = debug_publish,
                                   teq_nodes = teq_nodes,
                                   return_coords = True, 
                                   trainable = trainable,
                                   space_reg_strength = 1e-6 )
    
    x = Concatenate()([xgn, x])
    
    score = Dense(1, activation='sigmoid', name=name+'_score', trainable = trainable)(x)
    pre_processed['features'] = x #pass through
    if decouple_coords:
         #prime coordinates are passed through to here (see above)
        gn_coords = Dense(gn_coords.shape[1], name=name+'_coords', trainable = trainable, use_bias=False)(x)
    
    ud_graph = mini_tree_create(
        score,
        gn_coords,
        rs,

        t_idx, pre_processed['t_energy'], 

        is_track = is_track,

        K = 5,

        K_loss = 48,
        score_threshold=0.5,
        low_energy_cut = low_energy_cut,
        
        record_metrics = record_metrics,
        trainable = trainable,
        name = name+'_tree_creation',
        always_record_reduction = always_record_reduction
        )
    
    out = mini_tree_clustering(
        pre_processed,
        ud_graph,
        edge_dense = edge_dense,
        edge_pre_nodes = edge_pre_nodes,

        record_metrics = record_metrics,
        trainable = trainable,
        name = name+'_tree_clustering',
        produce_output = produce_output
        )
    
    return out, x

def post_tree_condensation_push(
        x_in, #before pushing as defined in the graph
        x_mix,
        graph,
        trainable = True,
        heads : int = 4,
        mix_nodes = 8,
        name = 'post_tree_push'):
    '''
    Defines a simple block to push up learnable quantities.
    The inputs x and x_mix are *before* the push up defined by the graph object.
    The output will have dimensionality *after* the push up.
    '''
    from GraphCondensationLayers import Mix
    
    x = x_in
    xup = []
    for i in range(heads):
        x_mix_i = Dense(mix_nodes, activation='elu', trainable = trainable,
              name = name + f'_int_mix_{i}')(x_mix)
        x_mix_i = Mix()(x_mix_i, graph)
        #sigmoid is ok, the 'mean' takes care of what the softmax does in classic attention
        attention = Dense(graph['nidx_down'].shape[1], activation='sigmoid', trainable = trainable,
              name = name + f'_weights_{i}')(x_mix_i)
        # add an epsilon to avoid numerical instabilities
        #this behaves very similar to standard attention now
        xm = PushUp(add_self=False, mode = 'mean')(x, graph, nweights = attention)
        xup.append(xm)
    
    if heads > 1:
        xup = Concatenate()(xup)
    else:
        xup = xup[0]
    return xup


def tree_condensation_block2(*args, **kwargs):
    #just define some defaults here
    return tree_condensation_block(*args, **kwargs,
                                   enc_nodes = 128,
                                   gn_nodes = 64,
                                   gn_neighbours = 128,
                                   teq_nodes = [64,64],
                                   edge_dense = [64,32],
                                   edge_pre_nodes = 32,
                                   low_energy_cut = 4.,
                                   name = 'tree_condensation_block2')

def double_tree_condensation_block(in_dict,
                             debug_outdir='', plot_debug_every=-1,
                             name = 'double_tree_condensation_block',
                             trainable = False,
                             record_metrics = False,
                             decouple_coords = False,
                             pre_gravnet = True,
                             debug_publish = None):
    
    if pre_gravnet: #run one single 'gravnet' to gather info about best coordinates; no need to learn coordinates yet so direct implementation

        nidx, dist = KNN(16, record_metrics=record_metrics, name='pre_knn_coords')([in_dict['prime_coords'], in_dict['row_splits']])

        xpre = Concatenate()([in_dict['prime_coords'], in_dict['features']])
        xpre = Dense(16, activation='tanh', name='pre_enc', trainable = trainable)(xpre)

        xscale = Dense(1, name='pre_scale', trainable = trainable)(xpre)
        dist = LocalDistanceScaling(name='pre_scale_dist', max_scale = 10.)([dist, xscale])
        if True:
            xgn = TranslationInvariantMP([16], 
                 layer_norm = True,
                 activation = None, #layer norm takes care
                 sum_weight = False,
                 name = name+'_pre_teqmp',
                 trainable = trainable)([xpre, nidx, dist ])

        else:
            xgn = DistanceWeightedMessagePassing([16], name=name+'_pre_dmp1', trainable = trainable)([xpre, nidx, dist])
        
        dist = StopGradient()(dist)
        in_dict['features'] = Concatenate()([xgn, in_dict['features'], dist])


    [out, graph], x_proc = tree_condensation_block(in_dict, 
                                  
                            #the latter overwrites the default arguments such that it is in training mode
                            debug_outdir=debug_outdir, plot_debug_every=plot_debug_every,
                            trainable = trainable,
                            record_metrics = record_metrics,
                            debug_publish = debug_publish,
                            decouple_coords = decouple_coords,
                            produce_output = True)
    
    ###########################################################################
    ### Just some debug out ###################################################
    ###########################################################################

    all_out = {'no_noise_idx_stage_0': graph['sel_idx_up']}
    
    in_dict['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = plot_debug_every,
                     name = 'dc_1st_stage',
                            publish = debug_publish,
                     outdir= debug_outdir )(in_dict['t_energy'], in_dict['t_idx'], graph)
    
    in_dict['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = plot_debug_every,
                     name = 'dc_1st_stage_no_tracks',
                            publish = debug_publish,
                     outdir= debug_outdir )(in_dict['t_energy'], in_dict['t_idx'], graph, is_track = in_dict['is_track'])
    
    #just to keep the plot in the loop
    graph['weights_down'] = DummyLayer()([graph['weights_down'], in_dict['t_energy']])

    ###########################################################################
    ### Second stage ##########################################################
    ########################################################################### 

    x = Concatenate()([x_proc, in_dict['features']])
    x = BatchNormalization(name = name + '_bn0', 
                                         trainable = trainable)(x) #this might be crucial after the summing
       
    xadd = post_tree_condensation_push(x, #pushed (with attention)
                                       x, # used to mix and build attention
                                       graph, 
                                       # take default here: heads = push_heads, 
                                       name = name+'_push',
                                       trainable = trainable) #this gets the original vector, adds more features to be pushed
    
    out['features'] = Concatenate()( [out['features'], xadd] )  
    out['features'] = BatchNormalization(name = name + '_bn1', 
                                         trainable = trainable)(out['features']) #this might be crucial after the summing

    [out2, graph2], x_proc2 = tree_condensation_block2(out, 
                                          debug_outdir=debug_outdir, plot_debug_every=plot_debug_every,
                                          debug_publish = debug_publish,
                                          trainable = trainable,
                                          decouple_coords = decouple_coords,
                                          record_metrics = record_metrics)
    
    
    ###########################################################################
    ### Done, now just checks and plots #######################################
    ###########################################################################

    #this is out and not out2 on purpose!
    out['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = plot_debug_every,
                     name = 'dc_2nd_stage',
                     publish = debug_publish,
                     outdir= debug_outdir )(out['t_energy'], out['t_idx'], graph2)
    
    out['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = plot_debug_every,
                     name = 'dc_2nd_stage_no_tracks',
                     publish = debug_publish,
                     outdir= debug_outdir )(out['t_energy'], out['t_idx'], graph2, is_track = out['is_track'])

    #make sure the above does not get optimised away
    graph2['weights_down'] = DummyLayer()([graph2['weights_down'], out['t_energy']])

    # create indices that were removed for tracking
    survived_both_stages = SelectUp()(graph['sel_idx_up'], graph2) 
    all_out['no_noise_idx_stage_1'] = graph2['sel_idx_up']
    all_out['survived_both_stages'] = survived_both_stages

    return out2, graph2, all_out
    

def tree_cleaning_block(pre_processed, # the full dictionary so that the truth can be fed through
                        coords, # the coordinates for the cleaning
                        score, # the score for the cleaning (sigmoid activated!)
                        score_threshold = 0.5, # high threshold -> more cleaning; between 0 and 1
                        name = 'tree_cleaning_block',
                        trainable = False
                        ):
    '''
    Contains the loss necessary for the tree cleaning
    '''
    
    ud_graph = mini_tree_create(
        score,
        coords,
        pre_processed['rs'],

        pre_processed['t_idx'], pre_processed['t_energy'], 

        is_track = pre_processed['is_track'],

        K = 5,

        K_loss = 48,
        score_threshold=score_threshold,
        
        record_metrics = trainable,
        trainable = trainable,
        name = name+'_tree_creation',
        cleaning_mode= True #!!!
        )



    out = mini_tree_clustering(
        pre_processed,
        ud_graph,
        edge_dense = [64,16],
        edge_pre_nodes = 32,

        record_metrics = trainable,
        trainable = trainable,
        name = name+'_tree_clustering',
        produce_output = True
        )
    
    return out


def tiny_pc_pool(
        orig_inputs,
        debug_outdir='',
        trainable=False,
        pass_through=False,
        name='pre_graph_pool',
        debugplots_after=-1,
        record_metrics=True,
        reduction_target = 0.05,
        K_loss = 48,
        low_energy_cut_target = 1.0,
        first_embed = True,
        coords = None,
        publish=None,
        dmp_steps=[8,8,8,8],
        dmp_compress = 32,
        K_nn = 16,
        K_gp = 5,
        is_second=False):
    '''
    This function needs pre-processed input (from condition_input)
    '''

    if pass_through:
        avail_keys = orig_inputs.keys()
        trans = GraphCondensation()

        if 'rs_down' in avail_keys:
            trans['rs_down'] = orig_inputs['rs_down']
        else:
            trans['rs_down'] = orig_inputs['row_splits']

        if 'rs_up' in avail_keys:
            trans['rs_up'] = orig_inputs['rs_up']
        else:
            trans['rs_up'] = orig_inputs['row_splits']

        zeros = ScalarMultiply(0.)(orig_inputs['features'][:,0:1])

        for k in ['nidx_down', 'distsq_down', 'sel_idx_up', 'weights_down']:
            if k in avail_keys:
                trans[k] = orig_inputs[k]
            else:
                trans[k] = zeros

        return trans, orig_inputs

    dwmp_activation = 'elu'

    if is_second:
        dmp_steps = [32,32,32,32]
        dmp_compress = 64
        low_energy_cut_target = 0.5
        K_nn = 32
        K_gp = 8
        first_embed = False
        dwmp_activation = 'tanh' #more smooth

    ## gather inputs and norm

    x = orig_inputs['features'] # coords
    if coords is None:
        coords = orig_inputs['prime_coords']
        coords = ScaledGooeyBatchNorm2(
            name = name+'_coords_batchnorm', trainable=trainable,
            fluidity_decay=1e-2,#can freeze quickly
            )(coords)

        coords = PlotCoordinates(plot_every=debugplots_after,
                                        outdir=debug_outdir,name=name+'_coords',
                                        publish=publish)(
                                            [coords,
                                             orig_inputs['rechit_energy'],
                                             orig_inputs['t_idx'],orig_inputs['row_splits']])

    rs = orig_inputs['row_splits']
    energy = orig_inputs['rechit_energy']
    is_track = orig_inputs['is_track']


    x = ScaledGooeyBatchNorm2(
            name=name+'_cond_batchnorm_a', trainable=trainable)([x, is_track])

    x = ScaledGooeyBatchNorm2(
            name=name+'_cond_batchnorm_b', trainable=trainable,
            invert_condition = True)([x, is_track])

    x_in = x
    x_emb = x
    #create different embeddings for tracks and hits
    if first_embed:

        x_track = Dense(16, activation='elu', name=name+'emb_xtrack',trainable=trainable)(x)
        x_hit = Dense(16, activation='elu', name=name+'emb_xhit',trainable=trainable)(x)
        x = MixWhere()([is_track, x_track, x_hit])
        x = ScaledGooeyBatchNorm2(name = name+'_emb_batchnorm', trainable=trainable,
                                  )(x)
        x_emb = x

    #simple gravnet
    nidx,dist = KNN(K=K_nn,record_metrics=record_metrics,name=name+'_np_knn',
                    min_bins=20)([coords, #stop gradient here as it's given explicitly below
                                  orig_inputs['row_splits']])#hard code it here, this is optimised given our datasets

    dist,nidx = SortAndSelectNeighbours(-1)([dist,nidx])#make it easy for edges

    dist = Sqrt()(dist)#make a stronger distance gradient for message passing

    #each is a simple attention head
    for i,n in enumerate(dmp_steps):
        #this is a super lightweight 'guess' attention; coords get explicit gradient down there

        x = DistanceWeightedMessagePassing([n],name=name+'np_dmp'+str(i),
                                        activation=dwmp_activation,#keep output in check
                                        exp_distances = True,
                                        trainable=trainable)([x,nidx,dist])# hops are rather light

        x = Dense(dmp_compress, activation='elu', name=name+'_pcp_x_out'+str(i),trainable=trainable)(x)

        x = ScaledGooeyBatchNorm2(
            name = name+'_dmp_batchnorm'+str(i),
            trainable=trainable,
            )(x)
        x = Concatenate()([x, x_emb])#skip connect


    score = Dense(1, activation='sigmoid',
                  name=name+'_pcp_score',
                  trainable=trainable)(x)

    score = LLGraphCondensationScore(
        name=name+'_ll_graph_condensation_score',
        record_metrics = record_metrics,
        K=K_loss,
        penalty_fraction = 0.5 + 0.3*(0.5 - reduction_target),
                active=trainable,
            low_energy_cut = low_energy_cut_target #allow everything below 2 GeV to be removed (other than tracks)
            )([score, coords, orig_inputs['t_idx'], orig_inputs['t_energy'], rs])

    trans_a = CreateGraphCondensation(
        trainable = trainable,
            reduction_target = reduction_target,
            K=K_gp,
            name=name+'_pcp_create',
            )(score,coords,rs, #nidx,dist_orig, #do not use nidx here yet.
              always_promote = is_track)

    x_e = Concatenate()([x,x_in])#skip
    x_e = Dense(16, activation='elu')(x_e)

    from GraphCondensationLayers import GetNeighbourMask, CreateGraphCondensationEdges2

    #this will determine how much feature / energy is pushed to 'up' vertices (or to /dev/null in case of noise)
    x_e = CreateGraphCondensationEdges2(name=name+'_gc_edges')(x_e, trans_a)# V x 5+1 x 15

    x_e = Dense(dmp_compress//2, name=name+'_graphn_edges_d1', activation = 'elu', trainable=trainable)(x_e)
    x_e = Dense(8, name=name+'_graphn_edges_d2', activation = 'elu', trainable=trainable)(x_e)
    # mix, they are distance ordered
    x_e = Flatten()(x_e)
    x_e = Concatenate()([x, x_e, trans_a['distsq_down']])
    x_e = Dense(12*(K_gp+1),name=name+'_graphn_edges_d3', activation = 'elu', trainable=trainable)(x_e)
    x_e = Dense(8*(K_gp+1),name=name+'_graphn_edges_d4', activation = 'elu', trainable=trainable)(x_e)
    x_e = Reshape([K_gp+1,8])(x_e)
    x_e = Dense(1, name=name+'_graphn_edges_final', activation = 'sigmoid', trainable=trainable)(x_e)
    x_e = Multiply()([x_e, GetNeighbourMask()(trans_a)]) #mask out non existing ones right away

    x_e = LLGraphCondensationEdges(
        name=name+'_ll_graph_condensation_edges',
        active=trainable,
        print_loss = trainable,
        record_metrics=record_metrics
        )(x_e, trans_a, orig_inputs['t_idx'])

    #this normalises the edges
    trans_a = InsertEdgesIntoTransition()(x_e, trans_a)

    trans_a = MLGraphCondensationMetrics(
        name = name + '_graphcondensation_metrics',
        record_metrics = record_metrics,
        )(trans_a, orig_inputs['t_idx'], orig_inputs['t_energy'],is_track=is_track)

    orig_inputs['t_energy'] = PlotGraphCondensationEfficiency(
                     plot_every = debugplots_after,
                     outdir= debug_outdir ,
                     name = name + '_efficiency',
                     publish = publish)(orig_inputs['t_energy'], orig_inputs['t_idx'], trans_a)

    out={}

    for k in orig_inputs.keys():
        if 't_' == k[0:2]:
            out[k] = SelectUp()(orig_inputs[k],trans_a)


    out['prime_coords'] = PushUp(add_self=True)(orig_inputs['prime_coords'],
                                                trans_a, weight = energy)

    out['rechit_energy'] = PushUp(mode='sum', add_self=True)(energy, trans_a)
    out['is_track'] = SelectUp()(is_track, trans_a)
    out['row_splits'] = trans_a['rs_up']

    out['coords'] = PushUp(add_self=True)(orig_inputs['coords'], trans_a, weight = energy)
    out['cond_coords'] = PushUp(add_self=True)(coords, trans_a, weight = energy)

    x_o = Concatenate()([x,x_in])
    out['down_features'] = x_o # this is for further "bouncing"
    out['up_features'] = SelectUp()(x_o, trans_a)

    out['select_prime_coords'] = SelectUp()(orig_inputs['prime_coords'], trans_a)

    x_of = PushUp()(x_o, trans_a)
    x_of2 = PushUp()(x_o, trans_a, weight = energy)
    out['features'] = Concatenate()([out['prime_coords'],out['up_features'],x_of, x_of2])

    out['cond_coords_down'] = coords #mostly for reference



    return trans_a, out




