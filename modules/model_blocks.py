
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization, Add, Multiply
from Layers import ExpMinusOne, CondensateToPseudoRS, RaggedSumAndScatter, FusedRaggedGravNetLinParse, VertexScatterer, FusedRaggedGravNetAggAtt
from DeepJetCore.DJCLayers import  StopGradient, SelectFeatures, ScalarMultiply

import tensorflow as tf


def indep_energy_block(x, ccoords, beta, x_row_splits):
    x = StopGradient()(x)
    ccoords = StopGradient()(ccoords)
    beta = StopGradient()(beta)
    feat=[x]
    
    sx, psrs, sids, asso_idx, belongs_to_prs = CondensateToPseudoRS(radius=0.8,  
                                                    soft=True, 
                                                    threshold=0.1)([x, ccoords, beta, x_row_splits])

    sx = Concatenate()([RaggedSumAndScatter()([sx, psrs, belongs_to_prs]) ,sx])                                            
    feat.append(VertexScatterer()([sx, sids, sx]))
    
    sx,_ = FusedRaggedGravNetLinParse(n_neighbours=128,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_prs')([sx, psrs])
                                 
    x = VertexScatterer()([sx, sids, sx])
    feat.append(x)
    x,_ = FusedRaggedGravNetLinParse(n_neighbours=128,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_last')([x, x_row_splits])
    feat.append(x)
    x = Concatenate()(feat)
    
    x = Dense(64, activation='elu',name="dense_last_enblock_1")(x)
    x = Dense(64, activation='elu',name="dense_last_enblock_2")(x)
    energy = Dense(1, activation=None,name="dense_enblock_final")(x)
    energy = energy #linear
    return energy

def indep_energy_block2(x, energy, ccoords, beta, x_row_splits, energy_proxy=None, stopxgrad=True):
    if stopxgrad:
        x = StopGradient()(x)
    energy = StopGradient()(energy)
    ccoords = StopGradient()(ccoords)
    beta = StopGradient()(beta)
    feat=[x]
    
    x = Dense(64, activation='elu',name="dense_last_start_enblock_1")(x)
    x = Dense(64, activation='elu',name="dense_last_start_enblock_2")(x)
    x = Concatenate()([energy,x])
    
    sx, psrs, sids, asso_idx, belongs_to_prs = CondensateToPseudoRS(radius=0.8,  
                                                    soft=True, 
                                                    threshold=0.2)([x, ccoords, beta, x_row_splits])
                                                    
    sx = Dense(128, activation='elu',name="dense_set_sum_input")(sx)
    sx = Dense(128, activation='elu',name="dense_set_sum_input_b")(sx)
    sx = Dense(128, activation='elu',name="dense_set_sum_input_c")(sx)
    
    #deep set like approach
    sx = Concatenate()([RaggedSumAndScatter()([sx, psrs, belongs_to_prs]) ,sx])
                                                   
    feat.append(VertexScatterer()([sx, sids, sx]))
    
    sx,_ = FusedRaggedGravNetLinParse(n_neighbours=128,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_prs')([sx, psrs])
                                 
    x = VertexScatterer()([sx, sids, sx])
    feat.append(x)
    x = Concatenate()([x,energy])
    x,_ = FusedRaggedGravNetAggAtt(n_neighbours=256,
                                 n_dimensions=4,
                                 n_filters=64,
                                 n_propagate=[32,32,32,32],
                                 name='gravnet_enblock_last')([x, x_row_splits, beta])
    feat.append(x)
    x = Concatenate()(feat)
    
    x = Dense(64, activation='elu',name="dense_last_enblock_1")(x)
    x = Dense(64, activation='elu',name="dense_last_enblock_2")(x)
    #energy = None

    energy = Dense(1, activation=None,name="predicted_energy")(x)
    
    return energy








def create_default_outputs(raw_inputs, x, x_row_splits, energy_block=True, n_ccoords=2, 
                           add_beta=None, add_beta_weight=0.2, use_e_proxy=False,
                           scale_exp_e=True):
    
    
    
    beta = None   
    if add_beta is not None:
        
        #the exact weighting can be learnt, but there has to be a positive correlation
        from tensorflow.keras.constraints import non_neg
        
        assert add_beta_weight < 1
        n_adds = float(len(add_beta))
        if isinstance(add_beta, list):
            add_beta = Concatenate()(add_beta)
            add_beta = ScalarMultiply(1./n_adds)(add_beta)
        add_beta = Dense(1, activation='sigmoid', name="predicted_add_beta", 
                         kernel_constraint=non_neg(), #maybe it figures it out...?
                         kernel_initializer='ones'
                         )(add_beta)
        
        #tf.print(add_beta)
        
        add_beta = ScalarMultiply(add_beta_weight)(add_beta)
        
        beta = Dense(1, activation='sigmoid', name="pre_predicted_beta")(x)
        beta = ScalarMultiply(1 - add_beta_weight)(beta)
        beta = Add(name="predicted_beta")([beta,add_beta])
        
    else:
        beta = Dense(1, activation='sigmoid', name="predicted_beta")(x)
        
    
    #x_raw = BatchNormalization(momentum=0.6,name="pre_ccoords_bn")(raw_inputs)
    #pre_ccoords = Dense(64, activation='elu',name="pre_ccords")(Concatenate()([x,x_raw]))
    ccoords = Dense(2, activation=None, name="predicted_ccoords")(x)
    if n_ccoords>2:
        ccoords = Concatenate()([ccoords,Dense(n_ccoords-2, activation=None, name="predicted_ccoords_add")(x)])
    
    xy = Dense(2, activation=None, name="predicted_positions",kernel_initializer='zeros')(x)
    t = Dense(1, activation=None, name="predicted_time",kernel_initializer='zeros')(x)
    t = ScalarMultiply(1e-9)(t)
    xyt = Concatenate()([xy,t])
    
    energy = None
    if energy_block:
        e_proxy=None
            
        energy = indep_energy_block2(x, SelectFeatures(0,1)(raw_inputs),ccoords, beta, x_row_splits, energy_proxy=e_proxy) 
    else:
        energy = Dense(1,activation=None)(x)
        if scale_exp_e:
            energy = ExpMinusOne(name='predicted_energy')(energy)
        else:
            energy = ScalarMultiply(100.)(energy)
        
        
    #(None, 9) (None, 1) (None, 1) (None, 3) (None, 2)
    print(raw_inputs.shape, beta.shape, energy.shape, xyt.shape, ccoords.shape)
    return Concatenate(name="predicted_final")([raw_inputs, beta, energy, xyt, ccoords])


def create_output_layers(x, x_row_splits, n_ccoords=2,
                           add_beta=None, add_beta_weight=0.2, use_e_proxy=False,
                           scale_exp_e=True, n_classes=0):
    beta = None
    if add_beta is not None:

        # the exact weighting can be learnt, but there has to be a positive correlation
        from tensorflow.keras.constraints import non_neg

        assert add_beta_weight < 1
        n_adds = float(len(add_beta))
        if isinstance(add_beta, list):
            add_beta = Concatenate()(add_beta)
            add_beta = ScalarMultiply(1. / n_adds)(add_beta)
        add_beta = Dense(1, activation='sigmoid', name="predicted_add_beta",
                         kernel_constraint=non_neg(),  # maybe it figures it out...?
                         kernel_initializer='ones'
                         )(add_beta)

        # tf.print(add_beta)

        add_beta = ScalarMultiply(add_beta_weight)(add_beta)

        beta = Dense(1, activation='sigmoid', name="pre_predicted_beta")(x)
        beta = ScalarMultiply(1 - add_beta_weight)(beta)
        beta = Add(name="predicted_beta")([beta, add_beta])

    else:
        beta = Dense(1, activation='sigmoid', name="predicted_beta")(x)

    # x_raw = BatchNormalization(momentum=0.6,name="pre_ccoords_bn")(raw_inputs)
    # pre_ccoords = Dense(64, activation='elu',name="pre_ccords")(Concatenate()([x,x_raw]))
    ccoords = Dense(2, activation=None, name="predicted_ccoords")(x)
    if n_ccoords > 2:
        ccoords = Concatenate()([ccoords, Dense(n_ccoords - 2, activation=None, name="predicted_ccoords_add")(x)])

    xy = Dense(2, activation=None, name="predicted_positions", kernel_initializer='zeros')(x)
    t = Dense(1, activation=None, name="predicted_time", kernel_initializer='zeros')(x)
    t = ScalarMultiply(1e-9)(t)
    xyt = Concatenate()([xy, t])

    energy = Dense(1, activation=None)(x)
    if scale_exp_e:
        energy = ExpMinusOne(name='predicted_energy')(energy)
    else:
        energy = ScalarMultiply(100.)(energy)

    if n_classes > 0:
        classes_scores = Dense(n_classes, activation=None, name="predicted_classification_scores")(x)
       
        return Concatenate(name="predicted_final")([beta, energy, xyt, ccoords, classes_scores])
    else:
        return Concatenate(name="predicted_final")([beta, energy, xyt, ccoords])


from datastructures import TrainData_OC
#new format!
def create_outputs(x, feat, energy=None, n_ccoords=3, n_classes=6, td=TrainData_OC(), add_features=True):
    '''
    returns pred_beta, pred_ccoords, pred_energy, pred_pos, pred_time, pred_id
    '''
    
    feat = td.createFeatureDict(feat)
    
    pred_beta = Dense(1, activation='sigmoid')(x)
    pred_ccoords = Dense(n_ccoords,use_bias=False,
                         kernel_initializer=tf.keras.initializers.orthogonal()
                         )(x) #bias has no effect
    pred_energy = Dense(1,kernel_initializer='zeros')(x)
    if energy is not None:
        pred_energy = Multiply()([pred_energy,energy])
        
    pred_pos =  Dense(2)(x)
    pred_time = ScalarMultiply(10.)(Dense(1)(x))
    if add_features:
        pred_pos =  Add()([feat['recHitXY'],pred_pos])
        pred_time = Add()([feat['recHitTime'],pred_time])
    pred_id = Dense(n_classes, activation="softmax")(x)
    
    return pred_beta, pred_ccoords, pred_energy, pred_pos, pred_time, pred_id





