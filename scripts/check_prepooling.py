import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator

datacollection = sys.argv[1]


dc = DataCollection(datacollection)
dataDir = dc.dataDir
samples = dc.samples


def closest_integer_product_larger_than(interger_input):
    """
    find two integers a and b such that
        a*b >= interger_inpu
        a*b is minimal
        |a - b| is minimal
    """

    sqrt = int(np.sqrt(interger_input))
    a = sqrt
    b = a + 1
    if a * b >= interger_input:
        return a, b
    else:
        return a + 1, b




for i, sample in enumerate(samples):
    if i > 0:
        break
    td = dc.dataclass()
    td.readFromFile(os.path.join(dataDir, sample))
    gen = TrainDataGenerator()
    gen.setBatchSize(1)
    gen.setSquaredElementsLimit(False)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)
    num_steps = gen.getNBatches()
    generator = gen.feedNumpyData()

    for j in range(num_steps):
        if j > 0:
            break
        data = next(generator)[0]
        allFeatures = td.interpretAllModelInputs(data)
        features = allFeatures['features']
        nrows, ncols = closest_integer_product_larger_than(features.shape[1])

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axs = axs.flatten()
        for k in range(features.shape[1]):
            axs[k].hist(features[:, k], bins=100, density=True)
            axs[k].set_title(str(k))

        plt.savefig(os.path.join(dataDir, f"file_{i}_event_{j}_features.png"))
        # save figure



