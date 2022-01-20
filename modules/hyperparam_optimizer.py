raise NotImplementedError('Needs to be revamped with the new code. To be done soon')

import time

from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
import matching_and_analysis



class OCHyperParamOptimizer():
    def __init__(self, analyzer, limit_n_endcaps=-1, database_manager=None, table_prefix='hyperparam_optimizer',
                 matching_type=matching_and_analysis.MATCHING_TYPE_MAX_FOUND,
                 beta_bounds=[0.,1.],
                 distance_bounds=[0.,1.]):
        self.limit_n_endcaps=limit_n_endcaps
        self.database_manager = database_manager
        self.table_prefix = table_prefix
        self.iteration = 0
        self.matching_type=matching_type
        self.analyzer = analyzer

        self.beta_bounds = beta_bounds
        self.distance_bounds = distance_bounds

    class Logger():
        def __init__(self, subscriber):
            self.iter = 0
            self.subscriber = subscriber

        def update(self, event, insta):
            current_max = insta.max

            if self.subscriber is not None:
                self.subscriber(self.iter, current_max["target"], current_max["params"])
            self.iter += 1


    def optimize(self, data, num_iterations, init_points=2, subscriber=None):
        """

        :param data: Data to optimize on
        :param num_iterations: Number of iterations
        :param init_points: Number of initial points to query before forming a model
        :param subscriber: a callback function, with 3 parameters, [iteration, max, params]
        :return:
        """
        self.data = data
        pbounds = {'d': (self.distance_bounds[0], self.distance_bounds[1]), 'b': (self.beta_bounds[0], self.beta_bounds[1])}
        optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=pbounds,
            random_state=1,
        )
        optimizer.subscribe(Events.OPTIMIZATION_STEP, OCHyperParamOptimizer.Logger(subscriber))
        optimizer.maximize(
            init_points=init_points,
            n_iter=num_iterations,
        )

        self.max_point = optimizer.max
        return optimizer.max

    def remove_data(self):
        del self.data

    def black_box_function(self, b, d):
        print("Running on ", b, d)
        graphs, metadata = self.analyzer.analyse_from_data(self.data, b, d, limit_endcaps=self.limit_n_endcaps)

        if self.database_manager is not None:
            plotter = HGCalAnalysisPlotter()
            plotter.add_data_from_analysed_graph_list(graphs, metadata)
            print("Writing to database")
            plotter.write_data_to_database(self.database_manager, self.table_prefix)
            print("Written")
            # print("Ran", b,d,target)

        target = metadata['reco_score']
        print("Target was ", target)

        return target


