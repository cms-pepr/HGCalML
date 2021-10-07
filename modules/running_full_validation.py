import gzip
import os
import pickle
import traceback
import gc

import mgzip
import numpy as np
import tensorflow as tf

import graph_functions
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
from plotting_callbacks import publish
from importlib import reload


graph_functions = reload(graph_functions)


class RunningFullValidation(tf.keras.callbacks.Callback):
    def __init__(self, after_n_batches, predictor, optimizer, database_manager, pdfs_path=None, min_batch = 0, table_prefix='gamma_full_validation',
                 run_optimization_loop_for=80, optimization_loop_num_init_points=5, trial_batch=10):
        """
        :param after_n_batches: Run this after number batches equal to this
        :param predictor: Predictor class for HGCal
        :param optimizer: Bayesian optimizer class for HGCal (OCHyperParamOptimizer)
        :param database_manager: Database writing manager, where to write the data, leave None without proper understanding
        of underlying database operations
        :param min_batch: Run full validation after x plots
        :param pdfs_path: Where to
        :param table_prefix: Adds this prefix to storage tables, leave None without proper understanding of underlying
        database operations
        """
        super().__init__()
        self.after_n_batches = after_n_batches
        self.min_batch = 0
        self.predictor = predictor
        self.optimizer = optimizer
        self.database_manager = database_manager
        self.table_prefix = table_prefix
        self.min_batch = min_batch
        self.pdfs_path = pdfs_path
        self.run_optimization_loop_for = run_optimization_loop_for
        self.optimization_loop_num_init_points = optimization_loop_num_init_points
        self.trial_batch = trial_batch

    def on_train_batch_end(self, batch, logs=None):
        if self.model.num_train_step < self.min_batch:
            return

        print("Iteration ", self.model.num_train_step)

        if self.model.num_train_step % self.after_n_batches==0 or self.model.num_train_step == self.trial_batch:
            pass
        else:
            return

        print("\n\nGonna run callback to do full validation...\n\n", self.model.num_train_step)

        try:
            all_data = self.predictor.predict(model=self.model, output_to_file=False)
        except FileNotFoundError:
            print("Model file not found. Will skip.")
            return


        max_point = self.optimizer.optimize(all_data, num_iterations=self.run_optimization_loop_for, init_points=self.optimization_loop_num_init_points)
        b = max_point['params']['b']
        d = max_point['params']['d']
        graphs, metadata = self.optimizer.analyzer.analyse_from_data(all_data, b, d)
        self.optimizer.remove_data() # To save memory


        plotter = HGCalAnalysisPlotter()
        tags = dict()
        tags['iteration'] = int(self.optimizer.iteration)
        tags['training_iteration'] = int(self.model.num_train_step)

        plotter.add_data_from_analysed_graph_list(graphs, metadata, additional_tags=tags)

        if self.database_manager is not None:
            plotter.write_data_to_database(self.database_manager, self.table_prefix)
            print("Written to database")
        if self.pdfs_path is not None:
            plotter.write_to_pdf(pdfpath=os.path.join(self.pdfs_path, 'validation_results_%d.pdf'%self.model.num_train_step))







