from datastructures import TrainData_Cocoa
from djcdata.dataPipeline import TrainDataGenerator
import numpy as np
import matplotlib.pyplot as plt

td = TrainData_Cocoa()

td.readFromFile('/work/friemer/hgcalml/trainingdata/singleQuarkJet_train.djctd')
gen = TrainDataGenerator()
gen.setBatchSize(1)
gen.setBuffer(td)
gen.setSkipTooLargeBatches(False)

data = next(gen.feedTrainData())#this is a dict, row splits can be ignored, this is per event
df= TrainData_Cocoa.createPandasDataFrame(data, 0)
print(df.keys())
# Index(['recHitEnergy', 'recHitEta', 'recHitHitR', 'recHitID',
#        'recHitLogEnergy', 'recHitR', 'recHitTheta', 'recHitTime', 'recHitX',
#        'recHitY', 'recHitZ', 't_rec_energy', 'truthHitAssignedEnergies',
#        'truthHitAssignedEta', 'truthHitAssignedPIDs', 'truthHitAssignedPhi',
#        'truthHitAssignedT', 'truthHitAssignedX', 'truthHitAssignedY',
#        'truthHitAssignedZ', 'truthHitAssignementIdx',
#        'truthHitFullyContainedFlag', 'truthHitSpectatorFlag'],
#       dtype='object')


phi_truth = np.arctan2(df['truthHitAssignedY'], df['truthHitAssignedX'])
eta_truth = df['truthHitAssignedZ']

# phi_truth = df['truthHitAssignedPhi']
# eta_truth = df['truthHitAssignedEta'] #Wrong definition of eta in the nanoML file vs cococa

# print(max(abs(phi_pred - phi_truth)))
# print(max(abs(eta_pred - eta_truth)))

phi_hit = np.arctan2(df['recHitY'], df['recHitX'])
#Calculate Eta from X,Y,Z
eta_hit = np.arcsinh(df['recHitZ']/np.sqrt(df['recHitX']**2 + df['recHitY']**2))

print(np.std(phi_hit - phi_truth))
print(max(abs(phi_hit - phi_truth)))
plt.hist(phi_hit - phi_truth, bins=25)
plt.ylabel('number of hits')
plt.xlabel('phi_hit - phi_truth')
plt.savefig('phi_hit.png')

print(np.std(eta_hit - eta_truth))
print(max(abs(eta_hit - eta_truth)))
plt.hist(eta_hit - eta_truth, bins=25)
plt.ylabel('number of hits')
plt.xlabel('eta_hit - eta_truth')
plt.savefig('eta_hit.png')



