
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization
from Layers import ExpMinusOne, CondensateToPseudoRS, RaggedSumAndScatter, FusedRaggedGravNetLinParse, VertexScatterer
from DeepJetCore.DJCLayers import  StopGradient, SelectFeatures, ScalarMultiply




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

def indep_energy_block2(x, energy, ccoords, beta, x_row_splits):
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
                                                    threshold=0.1)([x, ccoords, beta, x_row_splits])
                                                    
    sx = Dense(128, activation='elu',name="dense_set_sum_input")(sx)
    
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
    energy = Dense(1, activation=None,name="predicted_energy")(x)
    return energy








def create_default_outputs(raw_inputs, x, x_row_splits, energy_block=True, n_ccoords=2):
    
    beta = Dense(1, activation='sigmoid', name="predicted_beta")(x)
    
    x_raw = BatchNormalization(momentum=0.6,name="pre_ccoords_bn")(raw_inputs)
    pre_ccoords = Dense(64, activation='elu',name="pre_ccords")(Concatenate()([x,x_raw]))
    ccoords = Dense(n_ccoords, activation=None, name="predicted_ccoords")(pre_ccoords)
    
    xy = Dense(2, activation=None, name="predicted_positions",kernel_initializer='zeros')(x)
    t = Dense(1, activation=None, name="predicted_time",kernel_initializer='zeros')(x)
    t = ScalarMultiply(1e-9)(t)
    xyt = Concatenate()([xy,t])
    
    energy = None
    if energy_block:
        energy = indep_energy_block2(x, SelectFeatures(0,1)(raw_inputs),ccoords, beta, x_row_splits) 
    else:
        energy = Dense(1,activation=None)(x)
        energy = ExpMinusOne(name='predicted_energy')(energy)

    
    return Concatenate(name="predicted_final")([raw_inputs, beta, energy, xyt, ccoords])









