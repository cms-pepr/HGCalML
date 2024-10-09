"""
Helper script to run prediction and analysis for a cocoa model on both test sets.
"""

from DeepJetCore.wandb_interface import wandb_wrapper as wandb
wandb.active = False 

from argparse import ArgumentParser
from hgcal_predictor import HGCalPredictor
import os
from analyse_cocoa_predictions import analyse

if __name__ == '__main__':
    parser = ArgumentParser(
        'Helper script to run prediction and analysis for a cocoa model on both test sets.')
    parser.add_argument('modellocation',
        help='location of the KERAS_model.h5 file')
    parser.add_argument('outputdir',
        help="Output directory",
        default='')
    parser.add_argument('-testfiles',
        help="Location of a folder with a quarkDC._n.djcdc and gluonDC._n.djcdc test file",
        default='/work/friemer/hgcalml/testdata/')
    parser.add_argument("--max_files", help="Limit number of files", default=-1)
    parser.add_argument("--max_steps", help="Limit number of steps per file", default=-1)
    #Arguments for analyse
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-m', help='Matching mode', default='cocoa')
    parser.add_argument('--nfiles',
        help='Maximum number of files to analyse. -1 for everything in the preddir',
        default=-1)
    parser.add_argument('--nevents', help='Maximum number of events (per file) to analyse', default=-1)
    
    args = parser.parse_args()


HGCalPredictor(
        os.path.join(args.testfiles, 'quarkDC._n.djcdc'),
        os.path.join(args.testfiles, 'quarkDC._n.djcdc'),
        os.path.join(args.outputdir, 'predictionQuark'),
        inputdir=None,
        unbuffered=False,
        max_files=int(args.max_files),
        toydata=False,
        ).predict(model_path=args.modellocation, max_steps=int(args.max_steps))

HGCalPredictor(
        os.path.join(args.testfiles, 'gluonDC._n.djcdc'),
        os.path.join(args.testfiles, 'gluonDC._n.djcdc'),
        os.path.join(args.outputdir, 'predictionGluon'),
        inputdir=None,
        unbuffered=False,
        max_files=int(args.max_files),
        toydata=False,
        ).predict(model_path=args.modellocation, max_steps=int(args.max_steps))

analyse(preddir=os.path.join(args.outputdir, 'predictionQuark'),
        outputpath=os.path.join(args.outputdir, 'plotsQuark'),
        beta_threshold=float(args.b),
        distance_threshold=float(args.d),
        iou_threshold=float(args.i),
        matching_mode=args.m,
        analysisoutpath=os.path.join(args.outputdir, 'analysisQuark'),
        nfiles=int(args.nfiles),
        local_distance_scaling=True,
        de_e_cut=-1.0,
        angle_cut=-1.0,
        nevents=int(args.nevents),
        datasetname='Quark Jet',
        )

analyse(preddir=os.path.join(args.outputdir, 'predictionGluon'),
        outputpath=os.path.join(args.outputdir, 'plotsGluon'),
        beta_threshold=float(args.b),
        distance_threshold=float(args.d),
        iou_threshold=float(args.i),
        matching_mode=args.m,
        analysisoutpath=os.path.join(args.outputdir, 'analysisGluon'),
        nfiles=int(args.nfiles),
        local_distance_scaling=True,
        de_e_cut=-1.0,
        angle_cut=-1.0,
        nevents=int(args.nevents),
        datasetname='Gluon Jet',
        )