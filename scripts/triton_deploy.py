#!/usr/bin/env python3

from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('inputModel')
parser.add_argument('outputDir')
parser.add_argument('-d', help='distance threshold for object condensation (t_d in the paper)', default=0.7, type=float)
parser.add_argument('-b', help='beta threshold for object condensation (t_beta in the paper)', default=0.9, type=float)
parser.add_argument('-s', help='use soft condensation (similar to soft NMS)', action='store_true' , default=False )
args = parser.parse_args()

from DeepJetCore.modeltools import loadModelAndFixLayers
from DeepJetCore.DJCLayers import SelectFeatures
from index_dicts import feature_length
import tensorflow as tf
from condensate_op import BuildCondensates
import os
import numpy as np
from DeepJetCore.training.gpuTools import DJCSetGPUs
import time
from LayersRagged import Condensate
DJCSetGPUs()


        


hgcalml_path=os.getenv("DEEPJETCORE_SUBPACKAGE")
print('HGCalML in',hgcalml_path)

model_in = loadModelAndFixLayers(args.inputModel,"")

print('model_in inputs',model_in.inputs)

x,_ = model_in.outputs #we just need the first one
row_splits = model_in.inputs[1]
print('row_splits',row_splits.shape)
fullout = Condensate(args.d,args.b,args.s,feature_length=feature_length,name="output")([x,row_splits])
model = tf.keras.Model(inputs=model_in.inputs, outputs=fullout) 
model.compile()

print('model inputs',model.inputs)
print('model outputs',model.outputs)

print(model.summary())

print('inputs:')
for l in model.inputs:
    print('name:',l.name, 'dtype:',l.dtype,'shape',l.shape)
print('outputs:')
for l in model.outputs:
    print('name:',l.name, 'dtype:',l.dtype,'shape',l.shape)
    
model.save(args.inputModel[:-3]+'_extended.h5')    
print('running a test inference')

datadir = hgcalml_path+'/triton/oc_client/testdata/'
d = np.load(datadir+"np_0_feat_0.npy")
rs = np.load(datadir+"np_0_feat_1.npy")

start_time = time.time()
clusters, rs = model.predict([d,rs],batch_size=1000000)  
print('predict took', time.time()-start_time, 's')  
print(clusters.shape, rs.shape, rs) 
print('predicted betas', clusters[:,feature_length]) 
print('predicted energies', clusters[:,feature_length+1]) 


model.save(
    args.outputDir, overwrite=True, include_optimizer=False, save_format="tf",
    signatures=None, options=None
)    
