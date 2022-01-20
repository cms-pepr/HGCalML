
import os
import uuid

from OCHits2Showers import OCHits2Showers
from ShowersMatcher import ShowersMatcher
from hgcal_predictor import HGCalPredictor
from callbacks import RunningFullValidation
from callbacks import plotClusterSummary

def build_callbacks(train,
                    beta_and_dist_thresholds=[(0.1, 0.2),
                                              (0.1,0.5),
                                              (0.1,0.8),
                                              (0.1,1.0),
                                              (0.1,1.5),

                                              (0.3,0.2),
                                              (0.3,0.5),
                                              (0.3,0.8),
                                              (0.3,1.0),
                                              (0.3,1.5)],
                    iou_threshold=0.1,
                    matching_mode='iou_max',
                    local_distance_scaling=True,
                    is_soft=True,
                    de_e_cut=-1,
                    angle_cut=-1,
                    full_analysis_after_batches=7000,
                    cluster_summary_after_batches=400):

    if beta_and_dist_thresholds is not None:
        assert type(beta_and_dist_thresholds) == list
        assert len(beta_and_dist_thresholds) >= 1


    cb = []
    os.system('mkdir -p %s' % (train.outputDir + "/summary/"))

    if full_analysis_after_batches != -1:
        predictor = HGCalPredictor(os.path.join(train.outputDir, 'valsamples.djcdc'),
                                   os.path.join(train.outputDir, 'valsamples.djcdc'),
                                   os.path.join(train.outputDir, 'temp_val_outputs'), unbuffered=False,
                                   model_path=os.path.join(train.outputDir, 'KERAS_check_model_last_save'),
                                   inputdir=os.path.split(train.inputData)[0], max_files=1)

        hits2showers = OCHits2Showers(0.1, 0.1, is_soft, local_distance_scaling, op=True)
        showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

        os.system('mkdir -p %s/full_validation_plots' % (train.outputDir))

        cb += [RunningFullValidation(after_n_batches=full_analysis_after_batches,
                                    predictor=predictor,
                                    hits2showers=hits2showers,
                                    showers_matcher=showers_matcher,
                                    test_on_points=beta_and_dist_thresholds,
                                    pdfs_path=os.path.join(train.outputDir,
                                                            'full_validation_plots'),
                                    min_batch=0,
                                    limit_endcaps = -1,#all endcaps in file
                                    limit_endcaps_by_time = 180,#in seconds, don't spend more than 10 minutes on this
                                    trial_batch=10
        )]

    cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=cluster_summary_after_batches
        )
    ]
    
    print('HGCalML callbacks setup')

    return cb

