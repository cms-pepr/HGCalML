
from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint #, ReduceLROnPlateau # , TensorBoard
# loss per epoch
from time import time
from pdb import set_trace
import matplotlib
matplotlib.use('Agg') 
import wandb


class wandbCallback(Callback):
    # Just log everything to wandb
    def __init__(self):
        self.curr_epoch=0

    def _record_data(self,logs,record_epoch=None):
        #step_number = self.params['steps'] * self.epoch + self.params['batch']
        wandb.log(logs)
        #print("step_number", step_number, "logs", logs)
        # Also log separate metrics on ends of epochs
        if record_epoch is not None:
            wandb.log({"epoch_" + k: v for k, v in logs.items()}, step=self.curr_epoch)

    def on_batch_end(self,batch,logs={}):
        self._record_data(logs)

    def on_epoch_end(self,epoch,logs={}):
        self._record_data(logs)
        self.curr_epoch += 1
        self._record_data(logs,record_epoch=True)

