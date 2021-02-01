
'''
Define custom losses here and add them to the global_loss_list dict (important!)
Keep this file to functions with a few lines adding loss components
Do the actual implementation of the loss components in Loss_implementations.py

'''
global_loss_list = {}

import tensorflow as tf

##### prototype for merging loss function
from betaLosses import *
global_loss_list['obj_cond_loss_truth'] = obj_cond_loss_truth
global_loss_list['obj_cond_loss_rowsplits'] = obj_cond_loss_rowsplits
global_loss_list['pre_training_loss'] = pre_training_loss

global_loss_list['pretrain_obj_cond_loss_rowsplits'] = pretrain_obj_cond_loss_rowsplits
global_loss_list['pretrain_obj_cond_loss_truth'] = pretrain_obj_cond_loss_truth






####### for the 'pre'clustering tests
