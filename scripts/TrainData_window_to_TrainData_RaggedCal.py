from datastructures.TrainData_RaggedCal import TrainData_RaggedCal
import tensorflow as tf
import os

foldername = '/mnt/ceph/users/jkieseler/HGCalML_data/Sept2020_19_production_1x1/full/'
outputfoldername = '/mnt/home/sqasim/ceph/HGCalML_data/Nov2020_02_production/djc'


for _filename in os.listdir(foldername):
    print("Working on ", _filename)
    if _filename.startswith('dataCollection'):
        continue
    outfilename = os.path.join(outputfoldername, _filename)
    filename = os.path.join(foldername, _filename)
    td = TrainData_RaggedCal()
    td.convertAndWriteFromOldFiles(filename, outfilename)
    print("Done")
