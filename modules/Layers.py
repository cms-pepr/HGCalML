
from caloGraphNN_keras import weighted_sum_layer,GlobalExchange,GravNet,GarNet
# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

global_layers_list['GlobalExchange']=GlobalExchange
global_layers_list['GravNet']=GravNet
global_layers_list['GarNet']=GarNet
global_layers_list['weighted_sum_layer']=weighted_sum_layer

