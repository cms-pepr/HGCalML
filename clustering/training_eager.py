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
import argparse
import numpy as np

from object_condensation import remove_zero_length_elements_from_ragged_tensors, evaluate_loss

parser = argparse.ArgumentParser(description='Training session id')
parser.add_argument('--trainid', type=str,
                    default="")
parser.add_argument('--modeltype', type=str,
                    default="")
args = parser.parse_args()
should_overfit=True


if(args.modeltype==''):
    my_model = GravnetModelBeta()
elif (args.modeltype=='overfitting-king'):
    print("overfitting king")
    my_model = OverfittingKing()
elif (args.modeltype=='overfitting-queen'):
    print("overfitting queen")
    my_model = OverfittingQueen()
elif (args.modeltype=='overfitting-prince'):
    print("overfitting queen")
    my_model = OverfittingPrince()

print("Model object made now initializing the parameters etc with one call")

print("But... to do that, we first need to load data. So loading data. Gonna take a while.")

data = dc.DataCollection('/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/50_part_sample/testdata/dataCollection.djcdc')
data.setBatchSize(10000)
data.invokeGenerator()
nbatches = data.generator.getNBatches()
print("The data has",nbatches,"batches.")
gen = data.generatorFunction()

batch = xbatch = gen.next()


print(np.sum(batch[0][0]))




print("First call happening now")
my_model.call(xbatch[0][0], xbatch[0][1])
print("Done... all good now I guess")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
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
            classes, row_splits = ragged_constructor((batch[1][0][:, 0][..., tf.newaxis], row_splits))

            classes = classes[:, 0]
            row_splits = tf.convert_to_tensor(row_splits)
            row_splits = tf.cast(row_splits, tf.int32)
            row_splits = remove_zero_length_elements_from_ragged_tensors(row_splits)

            # print(clustering_space.shape, beta_values.shape, row_splits.shape, classes.shape)

            loss, losses = evaluate_loss(clustering_space, beta_values, classes, row_splits, Q_MIN=1, S_B=0.3)

            print("{}".format('%08d' % itx), "{}".format('%07.3F' % losses[0]), "{}".format('%07.3F' % losses[1]), "{}".format('%07.3F' % losses[2]), "{}".format('%07.3F' % losses[3]))

            beta_loss_first_term, beta_loss_second_term, attractive_loss, repulsive_loss = losses
        grads = tape.gradient(loss, my_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

        tf.summary.scalar("loss", loss, step=optimizer.iterations)
        tf.summary.scalar("beta loss first term", beta_loss_first_term, step=optimizer.iterations)
        tf.summary.scalar("beta loss second term", beta_loss_second_term, step=optimizer.iterations)
        tf.summary.scalar("répulsive loss", repulsive_loss, step=optimizer.iterations)
        tf.summary.scalar("attractive loss", attractive_loss, step=optimizer.iterations)

        if itx % 100 == 0:
            print("Saving model")
            manager.save()
            writer.flush()

