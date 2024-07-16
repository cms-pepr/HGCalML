



from DeepJetCore import TrainData
from DeepJetCore import SimpleArray
import numpy as np
import uproot3 as uproot

import os
import pickle
import gzip
import pandas as pd



class TrainData_imagePF(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
        #don't use for now
        #self.input_names=["input_hits","input_row_splits"]


    
    

    def fileIsValid(self, filename):
        True

    
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="SLCIOConverted"):
        '''
        convertDJCFromSource.py -c TrainData_imagePF -i my_file_list.txt -o <output dir>
        parallelises etc.
        output: 
         - output_dir/xxx.djctd ... 
         - output_dir/dataCollection.djcdc <- for the training
        '''
        
        #fileTimeOut(filename, 10)#10 seconds for eos to recover 

        farr = []


        # features: events x 50 x 50 x F

        # t_XYZ: arrays of (total events) x ?? , row splits (event offsets)

        # in the model:

        '''
         'recHitEnergy', <<< needed
         'recHitEta',
         'isTrack', <<< needed, all other optional
         'recHitTheta',
         'recHitR',
         'recHitX',
         'recHitY',
         'recHitZ',
         'recHitTime',
         'recHitHitR'


        x = inputs_features
        x = CNN2D()(...)
        x = CNN2D()(...)
        x = CNN2D()(...)
        x = CNN2D()(...)
        x = "flatten"(x) #flatten over events

        #from here on as in the other training scripts
        --> flat

        '''

        #t['t_idx'] = SimpleArray(np.array, rs=[0,2500,5000, ...])
        t={}

        # per hit: 
        return [#this is events x 2D x F
                farr, 
                #these are "flat" (events*hits) x X
                t['t_idx'],  # << index per particle , -1 for "noise"
                t['t_energy'], # << total true momentum
                t['t_pos'],  # << entry position, 2D; also zeros if you want
                t['t_time'], # << zero
                t['t_pid'],  # << PID -> zero
                t['t_spectator'],  # << all zero
                t['t_fully_contained'], # << all ones
                t['t_rec_energy'], # << total reconstructed energy sum / reco/smeared track momentum
                t['t_is_unique'] # just one hit per particle has a 1
                ],     [], []
    
    
    
    def createPandasDataFrame(self, eventno=-1):
        pass #later convenience function
    

    def interpretAllModelInputs(self, ilist, returndict=True):
        # to be filled properly

        if not returndict:
            raise ValueError('interpretAllModelInputs: Non-dict output is DEPRECATED. PLEASE REMOVE') 
        '''
        input: the full list of keras inputs
        returns: td
         - rechit feature array
         - t_idx
         - t_energy
         - t_pos
         - t_time
         - t_pid :             non hot-encoded pid
         - t_spectator :       spectator score, higher: further from shower core
         - t_fully_contained : fully contained in calorimeter, no 'scraping'
         - t_rec_energy :      the truth-associated deposited 
                               (and rechit calibrated) energy, including fractional assignments)
         - t_is_unique :       an index that is 1 for exactly one hit per truth shower
         - row_splits


         IMPOARTANT: for every SimpleArray you initialise with explicit row splits, the model will get the array content + row splits in the list
         
        '''
        out = {
            'features':ilist[0],
            'rechit_energy': ilist[0][:,0:1], #this is hacky. FIXME
            't_idx':ilist[2],
            't_energy':ilist[4],
            't_pos':ilist[6],
            't_time':ilist[8],
            't_pid':ilist[10],
            't_spectator':ilist[12],
            't_fully_contained':ilist[14],
            'row_splits':ilist[1]
            }
        #keep length check for compatibility
        if len(ilist)>16:
            out['t_rec_energy'] = ilist[16]
        if len(ilist)>18:
            out['t_is_unique'] = ilist[18]
        return out
    
     
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        pass #later
    
    
 