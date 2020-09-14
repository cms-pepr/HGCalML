#!/usr/bin/env python3

from argparse import ArgumentParser
import os

parser = ArgumentParser('')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
args = parser.parse_args()

inputdcfile = args.inputFile
outdir=args.outputDir+"/"
os.system('mkdir -p '+outdir)


from datastructures import TrainData_window
from DeepJetCore.compiled.c_simpleArray import simpleArray
from DeepJetCore.DataCollection import DataCollection

def replace(infile):

    td = TrainData_window()
    td.readFromFile(infile)
    
    feat,rs = td.transferFeatureListToNumpy()
    nrs = int(rs[-1])
    rs=rs[:nrs]
    truth,_ = td.transferTruthListToNumpy()
    
    
    farr = simpleArray()
    farr.createFromNumpy(feat, rs)
    
    truth[:,1] = truth[:,16]
    
    tarr = simpleArray()
    tarr.createFromNumpy(truth, rs)
    
    #tarr.cout()
    
    td_out = TrainData_window()
    td_out._store([farr],[tarr],[])
    
    td_out.writeToFile(outdir+infile)
    print(infile, 'done')
    
     
dc = DataCollection(inputdcfile)
inputdir = dc.dataDir
if not inputdir[:-1] == os.getcwd():
    print('needs to be called in same dir as dataCollection file',inputdir, os.getcwd())
    
inputdatafiles=[]    
for s in dc.samples:
    inputdatafiles.append(s)   
    

from multiprocessing import Pool
p = Pool()
res = p.map(replace, inputdatafiles) 
    
    
    
     
    
