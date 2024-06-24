from DeepJetCore.modeltools import load_model

from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
import imp
from Layers import *
from Losses import *
from Metrics import *


def get_custom_objects():
    imp.find_module('Layers')
    imp.find_module('Losses')
    imp.find_module('Metrics')
    custom_objs = {}
    custom_objs.update(djc_global_loss_list)
    custom_objs.update(djc_global_layers_list)
    custom_objs.update(global_loss_list)
    custom_objs.update(global_layers_list)
    custom_objs.update(global_metrics_list)
    return custom_objs    


custom_objs = get_custom_objects()

model_path = '/work/friemer/hgcalml/paper_BS100000__noDS_qm01_lr0005/model_results/KERAS_model.h5'


model_path = '/work/friemer/hgcalml/simplerGravNet5/results/KERAS_model.h5'


# import h5py

# with h5py.File(model_path, 'r') as f:
#     f.visititems(lambda name, obj: print(name, obj.shape if hasattr(obj, 'shape') else ''))


model = load_model(model_path)
model.save('testsavedMOdel.h5')
model2 = load_model('testsavedMOdel.h5')