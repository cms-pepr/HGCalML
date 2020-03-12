# -*- coding: utf-8 -*-

import tensorflow as tf
from LayersRagged import RaggedGravNet, RaggedConstructTensor
import DeepJetCore.DataCollection as dc
from model_gravnet_beta import GravnetModelBeta
import os
import uuid
import time
from model_overfitting_king import OverfittingKing
from model_overfitting_queen import OverfittingQueen
from model_overfitting_prince import OverfittingPrince
from model_dgcnn_alpha import DgcnnModelAlpha
from model_dgcnn_beta import DgcnnModelBeta
import argparse
import numpy as np
import h5py

from object_condensation import remove_zero_length_elements_from_ragged_tensors, object_condensation_loss
from segmentation_sota import SpatialEmbLossTf



def write_to_h5(x_data, clustering_space, beta_values, classes, row_splits):
    with h5py.File("train_data/checkfile.hdf5", "w") as outfile:
        outfile.create_dataset("x_data", data=x_data, dtype=np.float32)
        outfile.create_dataset("clustering_space", data=clustering_space, dtype=np.float32)
        outfile.create_dataset("beta_values", data=beta_values, dtype=np.float32)
        outfile.create_dataset("classes", data=classes, dtype=np.int32)
        outfile.create_dataset("row_splits", data=row_splits, dtype=np.int32)


parser = argparse.ArgumentParser(description='Training session id')
parser.add_argument('--trainid', type=str,
                    default="")
parser.add_argument('--modeltype', type=str,
                    default="")
parser.add_argument('--shouldoverfit', type=str,
                    default="True")
parser.add_argument('--lovasz', type=str,
                    default="False")
args = parser.parse_args()
should_overfit=True

should_overfit = (args.shouldoverfit == 'True' or args.shouldoverfit == 'true')
lovasz = (args.lovasz == 'True' or args.lovasz == 'true')

if (lovasz):
    lovasz_loss_calculator = SpatialEmbLossTf()

if(args.modeltype==''):
    my_model = GravnetModelBeta()
elif (args.modeltype=='overfitting-king'):
    print("overfitting king")
    my_model = OverfittingKing(clustering_dims=(5 if lovasz else 2))
elif (args.modeltype=='overfitting-queen'):
    print("overfitting queen")
    my_model = OverfittingQueen()
elif (args.modeltype=='overfitting-prince'):
    print("overfitting queen")
    my_model = OverfittingPrince()
elif (args.modeltype=='dgcnn-alpha'):
    print("dgcnn alpha")
    my_model = DgcnnModelAlpha(clustering_space_dimensions=(5 if lovasz else 2))
elif (args.modeltype=='dgcnn-beta'):
    print("dgcnn beta")
    my_model = DgcnnModelBeta()

print("Model object made now initializing the parameters etc with one call")

print("But... to do that, we first need to load data. So loading data. Gonna take a while.")


if should_overfit:
    data = dc.DataCollection('/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/50_part_sample/testdata/dataCollection.djcdc')
else:
    data = dc.DataCollection('/data/hgcal-0/store/jkiesele/50_part_sample_train/dataCollection.djcdc')

data.setBatchSize(10000 if should_overfit else 20000)
data.invokeGenerator()
nbatches = data.generator.getNBatches()
print("The data has",nbatches,"batches.")
gen = data.generatorFunction()

batch = xbatch = gen.next()


print(np.sum(batch[0][0]))




print("First call happening now - it will initialize weights")
c,_ = my_model.call(xbatch[0][0], xbatch[0][1])
print("Done...")


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
ragged_constructor = RaggedConstructTensor()

if len(args.trainid) == 0:
    training_number = uuid.uuid4().hex
else:
    training_number = str(args.trainid)
summaries_path = 'train_data/%s/summaries' % training_number
checkpoints_path = 'train_data/%s/checkpoints' % training_number

command = 'mkdir -p %s' % summaries_path
os.system(command)
command = 'mkdir -p %s' % checkpoints_path
os.system(command)


writer = tf.summary.create_file_writer(summaries_path)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=my_model)
manager = tf.train.CheckpointManager(
    checkpoint, directory=checkpoints_path, max_to_keep=5)
status = checkpoint.restore(manager.latest_checkpoint)

with writer.as_default():
    while True:
        if not should_overfit:
            try:
                batch = gen.next()
            except:
                data.generator.prepareNextEpoch()
                batch = gen.next()

        itx = int(optimizer.iterations.numpy())

        with tf.GradientTape() as tape:
            clustering_space, beta_values = my_model.call(batch[0][0], batch[0][1])

            row_splits = batch[0][1][:,0]
            input_ragged_trimmed,_ = ragged_constructor((batch[0][0], row_splits))
            classes, row_splits = ragged_constructor((batch[1][0][:, 0][..., tf.newaxis], row_splits))

            classes = classes[:, 0]
            row_splits = tf.convert_to_tensor(row_splits)
            row_splits = tf.cast(row_splits, tf.int32)
            row_splits = remove_zero_length_elements_from_ragged_tensors(row_splits)

            if lovasz:
                loss = lovasz_loss_calculator(row_splits, input_ragged_trimmed, clustering_space, beta_values, classes)
                print("Iteration", float(optimizer.iterations.numpy()), "Loss",float(loss.numpy()))
            else:
                loss, losses = object_condensation_loss(clustering_space, beta_values, classes, row_splits, Q_MIN=1, S_B=0.3)
                # loss, losses = evaluate_loss(clustering_space, beta_values, classes, row_splits, Q_MIN=1, S_B=0.3)

                print("{}".format('%08d' % itx), "{}".format('%07.3F' % losses[0]), "{}".format('%07.3F' % losses[1]), "{}".format('%07.3F' % losses[2]), "{}".format('%07.3F' % losses[3]))
                beta_loss_first_term, beta_loss_second_term, attractive_loss, repulsive_loss = losses
                tf.summary.scalar("beta loss first term", beta_loss_first_term, step=optimizer.iterations)
                tf.summary.scalar("beta loss second term", beta_loss_second_term, step=optimizer.iterations)
                tf.summary.scalar("répulsive loss", repulsive_loss, step=optimizer.iterations)
                tf.summary.scalar("attractive loss", attractive_loss, step=optimizer.iterations)

        tf.summary.scalar("loss", loss, step=optimizer.iterations)
        grads = tape.gradient(loss, my_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

        if itx % 100 == 0:
            print("Saving model")
            manager.save()
            writer.flush()

