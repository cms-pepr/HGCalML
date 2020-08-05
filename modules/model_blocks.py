
from tensorflow.keras.layers import Dense, Concatenate
from Layers import ExpMinusOne, CondensateToPseudoRS, RaggedSumAndScatter, FusedRaggedGravNetLinParse, VertexScatterer
from DeepJetCore.DJCLayers import  StopGradient

def indep_energy_block(x, ccoords, beta, x_row_splits):
    x = StopGradient()(x)
    feat=[x]
    
    sx, psrs, sids, asso_idx, belongs_to_prs = CondensateToPseudoRS(radius=0.8,  
                                                    soft=True, 
                                                    threshold=0.1)([x, ccoords, beta, x_row_splits])
    sx = RaggedSumAndScatter()([sx, psrs, belongs_to_prs])                                                
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