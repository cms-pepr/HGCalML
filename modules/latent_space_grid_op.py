import tensorflow as tf
from tensorflow.python.framework import ops



_lat_grid = tf.load_op_library('latent_space_grid.so')

def LatentSpaceGrid(coords, row_splits, min_cells=3, size=.8):
    '''
    
    .Attr("size: float")
    .Attr("min_cells: int") //same for all dimensions
    
    .Input("coords: float32")
    .Input("min_coord: float32") //per rs and dimension
    .Input("max_coord: float32")
    .Input("row_splits: int32")
    
    //these two resort
    .Output("select_idxs: int32") 
    .Output("pseudo_rowsplits: int32")
    
    //for reshaping for cnns
    .Output("n_cells_per_rs_coord: int32")
    
    //can be used to gather back to normal sorting
    .Output("vert_to_cell: int32");
    
    
    
    '''
    coords_ragged = tf.RaggedTensor.from_row_splits(
          values=coords,
          row_splits=row_splits)
    
    min = tf.reduce_min(coords_ragged,axis=1)
    max = tf.reduce_max(coords_ragged,axis=1)
                                       
    return _lat_grid.LatentSpaceGrid(size=size, 
                                     min_cells=min_cells,
                                     coords = coords,
                                     min_coord=min,
                                     max_coord=max,
                                     row_splits=row_splits )


@ops.RegisterGradient("LatentSpaceGrid")
def _LatentSpaceGrid(op, 
                       g_select_idxs,
                       g_pseudo_rowsplits,
                       g_n_cells_per_rs_coord,
                       g_vert_to_cell):
 

  return 4*[None] #no gradient for indices
  
  