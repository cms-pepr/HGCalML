import numpy as np
import tensorflow as tf
import datetime
import threading


print("MODULE OBSOLETE?",__name__)

class TensorBoardManager():
    def __init__(self, output_dir):

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = output_dir + '/' + current_time + '/'

        self.train_summary_writer = tf.summary.create_file_writer(output_dir)
        self.lock = threading.Lock()

    def step(self, step, dic):
        with self.lock:
            with self.train_summary_writer.as_default():
                for key, value in dic.items():
                    tf.summary.scalar(key, value, step)

