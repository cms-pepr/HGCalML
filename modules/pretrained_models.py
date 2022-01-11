
print("MODULE OBSOLETE?",__name__)
raise ImportError("MODULE",__name__,"will be removed")

from DeepJetCore.modeltools import load_model
import os

def get_model_path(modelname):
    
    models = os.getenv("DEEPJETCORE_SUBPACKAGE")+'/models/'
    
    if not os.path.isfile(models+modelname):
        raise ValueError("Model file "+models+modelname+" could not be found.")
    
    return models+modelname

def get_model(modelname):
    
    return load_model(get_model_path(modelname))
    