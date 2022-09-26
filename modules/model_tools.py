
import tensorflow as tf
from Layers import RobustModel,ExtendedMetricsModel
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.modeltools import apply_weights_where_possible
import os

def apply_weights_from_path(path_to_weight_model, existing_model):
    if not isinstance(existing_model, (RobustModel,ExtendedMetricsModel)):
        weightmodel = tf.keras.models.load_model(path_to_weight_model, custom_objects=get_custom_objects())
        existing_model = apply_weights_where_possible(existing_model, weightmodel)
        
        for le,lw in zip(existing_model.layers, weightmodel.layers):
            if hasattr(le, 'get_config'):
                if le.get_config() != lw.get_config():
                    ce = le.get_config()
                    cw = lw.get_config()
                    for cek in ce.keys():
                        if ce[cek] != cw[cek]:
                            print('warning: different configuration',le.name, ':', cek, ce[cek], cw[cek])
        
        return existing_model
    
    print('INFO: apply_weights_from_path: RobustModel deprecated ')
    newfilename=path_to_weight_model
    if str(newfilename).endswith('.h5'):
        newfilename = os.path.splitext(newfilename)[0] + '_save'
    
    weightmodel = tf.keras.models.load_model(newfilename, custom_objects=get_custom_objects())
    existing_model.model = apply_weights_where_possible(existing_model.model, weightmodel.model)
    
    
    return existing_model