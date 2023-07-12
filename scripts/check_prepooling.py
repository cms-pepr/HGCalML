import sys
import os
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator

datacollection = sys.argv[1]


dc = DataCollection(datacollection)
dataDir = dc.dataDir
samples = dc.samples


for i, sample in enumerate(samples):
    if i > 0:
        break
    td = dc.dataclass()
    td.readFromFile(os.path.join(dataDir, sample))
    gen = TrainDataGenerator()
    gen.setj/gen
    gen.setBatchSize(1)
    gen.setSquaredElementsLimit(False)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)
    num_steps = gen.getNBatches()
    generator = gen.feedNumpyData()

    for j in range(num_steps):
        data = next(generator)
        print(len(data))
        print(len(data[0]))
        break

