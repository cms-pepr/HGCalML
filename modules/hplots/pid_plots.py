import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve

class RocCurvesPlot():
    def __init__(self,classes=['EM', 'Hadronic', 'MIP', 'undef']):
        self.classes = classes
        self.primary_class = 0
        self.models_data = []
        self.dont_plot_idx=None

    def dont_plot(self, class_index):
        self.dont_plot_idx=class_index


    def _compute(self, true_id, pred_scores):
        data = dict()
        true_id = np.argmax(true_id, axis=1)

        data['curves'] = []

        for primary_class in range(len(self.classes)):
            for i in range(len(self.classes)):
                if primary_class == i:
                    continue
                filter = np.logical_or(np.equal(true_id, primary_class), np.equal(true_id, i))

                if np.sum(filter)!=0:

                    true_id_filtered = true_id[filter]
                    pred_scores_filtered = pred_scores[filter]

                    true_id_filtered = np.where(np.equal(true_id_filtered, primary_class), 1, 0)
                    pred_scores_filtered = pred_scores_filtered[:, primary_class][..., np.newaxis]

                    fpr, tpr,_ = roc_curve(true_id_filtered, pred_scores_filtered)
                else:
                    fpr, tpr = np.zeros(10, np.float), np.zeros(10, np.float)

                data[(primary_class, i)] = fpr, tpr

                data['curves'].append(
                    {
                        'fpr':fpr,
                        'tpr':tpr,
                        'primary_class':primary_class,
                        'secondary_class':i
                    }
                )

        return data

    def set_primary_class(self, primary_class):
        self.primary_class = primary_class

    def draw(self, name_tag_formatter=None):
        """
        :param name_tag_formatter: a function to which tags dict is given and it returns the name
        :return: figure
        """

        if len(self.models_data) > 1:
            print("Warning: Ignoring multiple models, this class only works with one model")

        if len(self.models_data) == 0:
            print("No model data found")
            return

        data = self.models_data[0]

        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

        # primary_class = data['curves'][0]['primary_class']

        # print("Drawing for", self.classes[primary_class])

        for curve in data['curves']:
            if curve['primary_class'] != self.primary_class:
                continue

            if self.dont_plot_idx is not None:
                if self.dont_plot_idx==curve['secondary_class']:
                    continue

            ax1.plot(curve['tpr'], 1-curve['fpr'], label='%s vs %s'% (self.classes[self.primary_class],self.classes[curve['secondary_class']]))

        ax1.legend()
        ax1.set_xlabel('Signal efficiency (TPR)')
        ax1.set_ylabel('Background rejection (TNR)')

        return fig


    def add_raw_values(self, true_id_one_hot, pred_scores, tags={}):
        """
        Will make one vs other roc curves in the same figure. primary class is the signal signal class.
        :return:
        """
        if type(true_id_one_hot) is not np.ndarray:
            raise ValueError("x values has to be numpy array")
        if type(pred_scores) is not np.ndarray:
            raise ValueError("y values has to be numpy array")

        data = self._compute(true_id_one_hot, pred_scores)
        data['tags'] = tags
        self.models_data.append(data)

    def write_to_database(self, database_manager, table_name):
        print("Warning: not implemented")
        return
        # print("Iterating")
        # for model_data in self.models_data:
        #     for curve in model_data['curves']:
        #         # print("Iterating xyz")
        #         fpr = curve['fpr']
        #         tpr = curve['tpr']
        #         primary_class = curve['primary_class']
        #         secondary_class = curve['secondary_class']
        #         tags = model_data['tags']
        #
        #         database_data = dict()
        #         for i in range(len(fpr)):
        #             database_data['fpr_%d' % i] = float(fpr[i])
        #             database_data['tpr_%d' % i] = float(tpr[i])
        #
        #         database_data['primary_class'] = primary_class
        #         database_data['secondary_class'] = secondary_class
        #
        #         for tag_name, tag_value in tags.items():
        #             database_data[tag_name] = tag_value
        #
        #         # print("Inserting ", database_data, table_name)
        #         database_manager.insert_experiment_data(table_name, database_data)

    def read_from_database(self, database_manager, table_name):
        print("Warning, not implemented")
        pass

    def add_processed_data(self, processed_data):
        self.models_data.append(processed_data)


class ConfusionMatrixPlot():
    def __init__(self, classes=['EM', 'Hadronic', 'MIP', 'undef']):
        self.classes = classes
        self.models_data = []
        self.dont_plot_idx=None

    def _compute(self, true_id, pred_id):
        data = dict()
        a = confusion_matrix(true_id, pred_id)
        result = np.zeros((len(self.classes),len(self.classes)), np.int32)
        result[:a.shape[0], :a.shape[1]] = a
        data['confusion_matrix'] = result


        return data

    def add_raw_values(self, true_id, pred_id, tags={}):
        if type(true_id) is not np.ndarray:
            raise ValueError("x values has to be numpy array")
        if type(pred_id) is not np.ndarray:
            raise ValueError("y values has to be numpy array")

        data = self._compute(true_id, pred_id)
        data['tags'] = tags
        self.models_data.append(data)

    def add_processed_data(self, processed_data):
        self.models_data.append(processed_data)

    def dont_plot(self, class_index):
        self.dont_plot_idx=class_index

    def draw(self, name_tag_formatter=None):
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

        if len(self.models_data) > 1:
            print("Warning: Ignoring multiple models, this class only works with one model")

        if len(self.models_data) == 0:
            print("No model data found")
            return

        data = self.models_data[0]

        confusion_matrix = data['confusion_matrix']
        classes = self.classes

        if self.dont_plot_idx is not None:
            confusion_matrix = np.delete(confusion_matrix, axis=0, obj=self.dont_plot_idx)
            confusion_matrix = np.delete(confusion_matrix, axis=1, obj=self.dont_plot_idx)
            classes = [c for i, c in enumerate(classes) if i!=self.dont_plot_idx]



        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                      display_labels=classes)

        disp.plot(ax=ax1)

        return fig

    def write_to_database(self, database_manager, table_name):
        # print("Iterating")
        for model_data in self.models_data:
            # print("Iterating xyz")
            confusion_matrix = model_data['confusion_matrix']
            confusion_matrix = np.reshape(confusion_matrix, (-1, ))
            tags = model_data['tags']

            if 'error' in model_data:
                error = model_data['error']

            database_data = dict()
            for i in range(len(confusion_matrix)):
                # database_data['bin_lower_energy'] = lows[i]
                # database_data['bin_upper_energy'] = highs[i]
                database_data['confusion_%d'%i] = float(confusion_matrix[i])

            for tag_name, tag_value in tags.items():
                database_data[tag_name] = tag_value

            # print("Inserting ", database_data, table_name)
            database_manager.insert_experiment_data(table_name, database_data)

    def get_tags(self):
        return [x['tags'] for x in self.models_data]

    def read_from_database(self, database_reading_manager, table_name, experiment_name=None, condition=None):
        results_dict = database_reading_manager.get_data(table_name, experiment_name, condition_string=condition)
        num_rows = len(results_dict['experiment_name'])

        results_dict_copy = results_dict.copy()

        for i in range(len(self.classes)**2):
            results_dict_copy.pop('confusion_%d' % i)

        tags_names = results_dict_copy.keys()

        # print(results_dict.keys())

        for row in range(num_rows):
            processed_data = dict()

            confusion = []
            for i in range(len(self.e_bins) - 1):
                confusion.append(float(results_dict['confusion_%d'%i][row]))

            confusion_matrix = np.reshape(confusion, (len(self.classes), len(self.classes)))

            processed_data['confusion_matrix'] = confusion_matrix

            tags = dict()

            for tag_name in tags_names:
                tags[tag_name] = results_dict[tag_name][row]

            processed_data['tags'] = tags
            self.add_processed_data(processed_data)


