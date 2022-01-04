
import tensorflow as tf
from Layers import RobustModel,ExtendedMetricsModel
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.modeltools import apply_weights_where_possible
import os

def apply_weights_from_path(path_to_weight_model, existing_model):
    if not isinstance(existing_model, (RobustModel,ExtendedMetricsModel)):
        weightmodel = tf.keras.models.load_model(path_to_weight_model, custom_objects=get_custom_objects())
        existing_model = apply_weights_where_possible(existing_model, weightmodel)
        return existing_model
    
    print('INFO: apply_weights_from_path: RobustModel deprecated ')
    newfilename=path_to_weight_model
    if str(newfilename).endswith('.h5'):
        newfilename = os.path.splitext(newfilename)[0] + '_save'
    
    weightmodel = tf.keras.models.load_model(newfilename, custom_objects=get_custom_objects())
    existing_model.model = apply_weights_where_possible(existing_model.model, weightmodel.model)
    return existing_model