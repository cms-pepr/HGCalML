import sql_credentials
from experiment_database_manager import ExperimentDatabaseManager
from experiment_database_reading_manager import ExperimentDatabaseReadingManager
import os
import matching_and_analysis
import uuid

from hgcal_predictor import HGCalPredictor
from hyperparam_optimizer import OCHyperParamOptimizer
from running_full_validation import RunningFullValidation
from running_plots import RunningMetricsDatabaseAdditionCallback, RunningMetricsPlotterCallback
from plotting_callbacks import plotClusterSummary

def build_callbacks(train, running_plots_beta_threshold=0.2, running_plots_distance_threshold=0.5,
                    running_plots_iou_threshold=0.1,
                    running_plots_matching_type=matching_and_analysis.MATCHING_TYPE_IOU_MAX,
                    running_plots_write_after_iterations=200, full_analysis_num_hyperparam_optimization_iterations=-1,
                    test_on_points=[(0.2,0.2), (0.8,0.8),(0.2,0.8),(0.8,0.2),(0.5,0.5)],
                    full_analysis_num_hyperparam_optimization_endcaps=-1, should_write_to_file=True,
                    should_write_to_remote_server=False, full_analysis_after_batches=5000,
                    running_plots_energy_gather_type=matching_and_analysis.ENERGY_GATHER_TYPE_CORRECTION_FACTOR_FROM_CONDENSATION_POINT):
    """

    This function will add two types of call backs:
    1. Running plots
    2. Full analysis plots

    Running plots include scalar metrics including: efficiency, mean response, and fake rate. These are computed in the
    background after every training iteration. Note that these might not be computed after every iteration strictly,
    if previous computation takes too much time, it can skip a training iteration. After
    running_plots_write_after_iterations number of training iterations, an HTML file is produced with plots as function
    of training iteration. They are useful for monitoring how the training is going.

    Full analysis plots are series of performance analysis plots made as a pdf file. They are made after
    full_analysis_after_batches number of training iterations. These take time to compute. Before computing them,
    bayesian hyper parameter optimization is done to find optimal values of beta and distance threshold. And then the
    performance analysis plots are made based on that.

    Full analysis plots include efficiency as function of truth shower energy, fake rate os function of predicted shower
    energy and much more.

    :param train: HGCalTraining instance (training_base)
    :param td: TrainData_NanoML or likewise
    :param running_plots_beta_threshold: beta threshold for live efficiency, response etc monitoring
    :param running_plots_distance_threshold: distance threshold for live efficiency, response etc monitoring
    :param running_plots_iou_threshold: iou threshold live for efficiency, response etc monitoring
    :param running_plots_matching_type: matching type (search for "Matching types" in in modules/matching_and_analysis.py)
    :param running_plots_write_after_iterations: Make html plots showing
    :param test_on_points: Run on these points instead of doing hyper param optimization, set to none for hyper param
           optimization
    :param full_analysis_after_batches: Run full analysis after this many batches, it will try to optimize wrt beta
           and distance threshold and won't use the parameters of this function and then produce a pdf file. Set to -1
           if full analysis plots are not required.
    :param full_analysis_num_hyperparam_optimization_iterations: Number of iterations for bayesian optimization for
           full analysis, set -1 for no hyper param optimization.
    :param full_analysis_num_hyperparam_optimization_endcaps: Number of endcaps for bayesian optimization for full
           analysis, set -1 for no hyper param optimization
    :param should_write_to_file: Keep to True otherwise can't produce running plots
    :param should_write_to_remote_server: Keep to False unless you want to make plots on other machines. Loss and
           efficiency plots will be made regardless after
    :return: a list of callbacks you can append to train function
    """

    if full_analysis_num_hyperparam_optimization_endcaps==-1 and full_analysis_num_hyperparam_optimization_endcaps !=-1:
        raise RuntimeError("Can't run hyper param optimization without setting full_analysis_num"
                           "hyperparam_optimization_endcaps and full_analysis_num_hyperparam_optimization_endcaps")
    if full_analysis_num_hyperparam_optimization_endcaps==-1 != test_on_points is None: # Just an xor
        raise RuntimeError("Either set test on points or hyper param optimization.")

    if test_on_points is not None:
        assert type(test_on_points) == list
        assert len(test_on_points) >= 1


    cb = []
    os.system('mkdir -p %s' % (train.outputDir + "/summary/"))
    
    td = train.train_data.dataclass()

    unique_id_path = os.path.join(train.outputDir, 'unique_id.txt')
    if os.path.exists(unique_id_path):
        with open(unique_id_path, 'r') as f:
            unique_id = f.readlines()[0].strip()
    else:
        unique_id = str(uuid.uuid4())[:8]
        with open(unique_id_path, 'w') as f:
            f.write(unique_id + '\n')

    database_manager = ExperimentDatabaseManager(mysql_credentials=(sql_credentials.credentials if should_write_to_remote_server else None),
                                                 file=(os.path.join(train.outputDir, "training_metrics.db") if should_write_to_file else None),
                                                 cache_size=100)

    database_reading_manager = ExperimentDatabaseReadingManager(
        file=os.path.join(train.outputDir, "training_metrics.db"))
    database_manager.set_experiment(unique_id)
    metadata = matching_and_analysis.build_metadeta_dict(beta_threshold=running_plots_beta_threshold,
                                                         distance_threshold=running_plots_distance_threshold,
                                                         iou_threshold=running_plots_iou_threshold,
                                                         matching_type=running_plots_matching_type,
                                                         energy_gather_type=running_plots_energy_gather_type)
    analyzer = matching_and_analysis.OCAnlayzerWrapper(metadata)
    cb += [RunningMetricsDatabaseAdditionCallback(td, database_manager=database_manager,
                                                  analyzer=analyzer)]
    cb += [RunningMetricsPlotterCallback(after_n_batches=running_plots_write_after_iterations, database_reading_manager=database_reading_manager,
                                         output_html_location=os.path.join(train.outputDir, "training_metrics.html"),
                                         publish=None)]

    if full_analysis_after_batches != -1:
        predictor = HGCalPredictor(os.path.join(train.outputDir, 'valsamples.djcdc'),
                                   os.path.join(train.outputDir, 'valsamples.djcdc'),
                                   os.path.join(train.outputDir, 'temp_val_outputs'), batch_size=1, unbuffered=False,
                                   model_path=os.path.join(train.outputDir, 'KERAS_check_model_last_save'),
                                   inputdir=os.path.split(train.inputData)[0], max_files=1)

        analyzer2 = matching_and_analysis.OCAnlayzerWrapper(
            metadata)  # Use another analyzer here to be safe since it will run scan on
        # on beta and distance threshold which might mess up settings

        if full_analysis_num_hyperparam_optimization_iterations != -1:
            optimizer = OCHyperParamOptimizer(analyzer=analyzer2,
                                              limit_n_endcaps=full_analysis_num_hyperparam_optimization_endcaps,
                                              beta_bounds=[0.05,1],
                                              distance_bounds=[0.05,1])
        else:
            optimizer = None



        os.system('mkdir %s/full_validation_plots' % (train.outputDir))
        cb += [RunningFullValidation(after_n_batches=full_analysis_after_batches, predictor=predictor, analyzer=analyzer2,
                                     optimizer=optimizer, test_on_points=test_on_points,
                                     database_manager=database_manager,
                                     pdfs_path=os.path.join(train.outputDir,
                                                            'full_validation_plots'), min_batch=8,
                                     run_optimization_loop_for=full_analysis_num_hyperparam_optimization_iterations,
                                     optimization_loop_num_init_points=5, trial_batch=10)]
    cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=800
        )
    ]

    return cb

