import tensorflow as tf
import DeepJetCore.DataCollection as dc
from LayersRagged import RaggedGravNet, RaggedConstructTensor
import numpy as np



data = dc.DataCollection('/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/50_part_sample/traindata/dataCollection.djcdc')
data.setBatchSize(40000)
data.invokeGenerator()
nbatches = data.generator.getNBatches()
print("The data has",nbatches,"batches.")
gen = data.generatorFunction()

ragged_constructor = RaggedConstructTensor()


num_unique_global = []
shower_sizes_global = []

for i in range(10000):
    print("Working on batch ", i)
    batch = gen.next()
    row_splits = batch[0][1][:, 0]


    classes, row_splits = ragged_constructor((batch[1][0][:, 0][..., tf.newaxis], row_splits))
    classes = tf.cast(classes[:,0], tf.int32)
    num_unique = []
    shower_sizes = []
    for i in range(len(row_splits)-1):
        classes_this_segment = classes[row_splits[i]:row_splits[i+1]]
        num_unique += [int(len(tf.unique(classes_this_segment)[0].numpy()))]
        shower_sizes += [int(row_splits[i+1] - row_splits[i])]
    print(num_unique)
    print(shower_sizes)
    num_unique_global += num_unique
    shower_sizes_global += shower_sizes


print(shower_sizes_global)

with open('showeres_sizes.txt', 'w') as f:
    for item in shower_sizes_global:
        f.write("%s\n" % item)
#
# x = np.histogram(num_unique_global, [x for x in range(100)])
#
# print(repr(x))
#


