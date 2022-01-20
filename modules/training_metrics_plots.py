import os
import pickle
import numpy as np
import mgzip
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

print("MODULE OBSOLETE?",__name__)

_debug = True
class TrainingMetricPlots():
    def __init__(self, reading_manager, experiment_name, ignore_cache, cache_path=None, database_table_name = 'training_performance_metrics_extended',
                 metrics=['loss','efficiency','fake_rate','sum_response','response','f_score_energy','num_pred_showers','num_truth_showers'],
                 titles=['Loss', 'Efficiency', 'Fake rate', 'Sum response', 'Response', 'F score energy', 'Num pred showers', 'Num Truth Showers']):
        self.reading_manager = reading_manager
        self.experiment_name = experiment_name
        self.ignore_cache = ignore_cache
        self.training_performance_metrics = None
        self.metrics = metrics
        self.titles = titles
        self.cache_path = 'training_metrics_plotter.cache' if cache_path is None else cache_path
        self.database_table_name = database_table_name

    def _combine(self, prev_data, new_data):
        combined_data = prev_data.copy()
        for key, value in prev_data.items():
            combined_data[key] = combined_data[key] + new_data[key]
        return combined_data

    def do_plot_to_html(self, html_location, average_over=1):
        self._fetch()

        nrows = math.ceil(len(self.metrics) / 2.)

        if _debug:
            print("Doing plots now")
        fig = make_subplots(rows=nrows, cols=2, subplot_titles=self.titles)

        for i, m in enumerate(self.metrics):
            self._plot_metric(fig, m, average_over, row=math.floor(i/2)+1, col=i%2+1)

            if _debug:
                print("Plot %d done" % i)

        # self._plot_metric(fig, 'loss', average_over,row=1, col=1)
        # self._plot_metric(fig, 'efficiency', average_over,row=2, col=1)
        # self._plot_metric(fig, 'fake_rate', average_over,row=3, col=1)
        # self._plot_metric(fig, 'sum_response', average_over,row=4, col=1)
        # self._plot_metric(fig, 'response', average_over,row=5, col=1)
        #
        # self._plot_metric(fig, 'f_score_energy', average_over,row=6, col=1)
        # self._plot_metric(fig, 'num_pred_showers', average_over,row=7, col=1)
        # self._plot_metric(fig, 'num_truth_showers', average_over,row=8, col=1)

        fig.update_layout(height=1200, width=1200, title_text="Training")

        if _debug:
            print("Writing HTML")
        fig.write_html(html_location)
        if _debug:
            print("HTML written")

    def running_mean(self, x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    def _plot_metric(self, figure, metric, average_over, row, col):
        experiment_names = np.unique(self.training_performance_metrics['experiment_name']).tolist()
        # print(experiment_names)

        for exp_name in experiment_names:
            self.training_performance_metrics[metric] = [float(x) for x in self.training_performance_metrics[metric]]
            y_values = self.training_performance_metrics[metric]
            filter = np.char.equal(self.training_performance_metrics['experiment_name'],exp_name)

            # print(type(filter))
            y_values = np.array(y_values)[filter]
            if average_over < len(y_values):
                if average_over > 1:
                    y_values = self.running_mean(y_values, average_over)

            trace = go.Scatter(x=np.array(self.training_performance_metrics['iteration'])[filter], y=y_values, name=exp_name)
            figure.add_trace(trace, row=row, col=col)

        # fig.add_trace(px.line(x=training_performance_metrics['iteration'], y=y_values))

    class ExperimentNotFoundError(RuntimeError):
        pass

    def _fetch(self):
        training_performance_metrics = None
        if not self.ignore_cache:
            if os.path.exists(self.cache_path):
                with mgzip.open(self.cache_path, 'rb') as f:
                    dumping_data = pickle.load(f)
                    print(dumping_data['experiment_name'])
                    if dumping_data['experiment_name'] == self.experiment_name:
                        training_performance_metrics = dumping_data['data']
                        print("Loaded data from cache...")
                    else:
                        print("Cache doesn't contain this experiment, will have to re-fetch.")

        condition_string = None
        if training_performance_metrics is not None:
            old_exp_names = np.unique(training_performance_metrics['experiment_name']).tolist()
            old_max_iterations = [np.max(np.array(training_performance_metrics['iteration'])[np.char.equal(training_performance_metrics['experiment_name'],expn)]) for expn in old_exp_names]

            condition_string = '(%s)' % ' OR '.join(["(experiment_name='%s' and iteration > '%d')"%(exp_n, iteration) for exp_n, iteration in zip(old_exp_names, old_max_iterations)])
            # condition_string = '(%s)' % condition_string


        if self.experiment_name is not None:
            experiment_name = str(self.experiment_name).split(',')
            if len(experiment_name) == 1:
                experiment_name = experiment_name[0]
        else:
            experiment_name = self.experiment_name

        if _debug:
            print("Going to fetch from server")
        new_data = self.reading_manager.get_data('%s' % self.database_table_name, experiment_names=experiment_name,
                                    condition_string=condition_string)
        if _debug:
            print("Fetch from server complete")

        if new_data is not None and training_performance_metrics is not None:
            training_performance_metrics = self._combine(training_performance_metrics, new_data)
        elif new_data is not None and training_performance_metrics is None:
            training_performance_metrics = new_data
        if not self.ignore_cache:
            with mgzip.open(self.cache_path,'wb') as f:
                dumping_data = {'experiment_name':self.experiment_name, 'data':training_performance_metrics}
                pickle.dump(dumping_data, f)

        if training_performance_metrics is None:
            print("Experiment not found, in your configured database, the following experiments were found:")


            available_experiment_names = self.reading_manager.get_data_from_query('SELECT DISTINCT(experiment_name) FROM %s'%self.database_table_name)
            available_experiment_names = [x[0] for x in available_experiment_names]
            print(available_experiment_names)
            raise TrainingMetricPlots.ExperimentNotFoundError("Experiment not found in your configured database")

        self.training_performance_metrics = training_performance_metrics
