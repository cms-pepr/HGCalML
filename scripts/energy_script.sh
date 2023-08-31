#!/bin/bash

# Script to run analysis for energy regression

DATA_ELECTRONS='/mnt/ceph/users/pzehetner/Paper/Test/test_energy_electrons/dataCollection.djcdc'
DATA_KAONS='/mnt/ceph/users/pzehetner/Paper/Test/test_energy_kaons/dataCollection.djcdc'
DATA_PHOTONS='/mnt/ceph/users/pzehetner/Paper/Test/test_energy_photons/dataCollection.djcdc'
DATA_PIONS='/mnt/ceph/users/pzehetner/Paper/Test/test_energy_pions/dataCollection.djcdc'

MODEL=$1
OUTPUT_DIR=$2

OUTPUT_ELECTRONS=$OUTPUT_DIR/electrons
OUTPUT_KAONS=$OUTPUT_DIR/kaons
OUTPUT_PHOTONS=$OUTPUT_DIR/photons
OUTPUT_PIONS=$OUTPUT_DIR/pions

if [ -z "$HGCALML" ]
then
    echo "Variable \$HGCALML is not set. Please set it by sourcing env.sh in the repository"
    exit 1
fi

if [ ! -f "$MODEL" ]
then
    echo "Model $MODEL does not exist. Please choose another model."
    exit 1
fi

if [ -d "$OUTPUT_DIR" ]
then
    echo "Directory $OUTPUT_DIR already exists. Please choose another directory."
    exit 1
else
    mkdir $OUTPUT_DIR
fi

if [ -z "$3" ]
then
    echo "No third argument given. Using default value of 0."
    N_FILES=-1
else
    N_FILES=$3
fi

cd $HOME

LOGFILE=$OUTPUT_DIR/energy_script.log
touch $LOGFILE

echo "Starting energy regression script" >> $LOGFILE
echo "Using model $MODEL" >> $LOGFILE
echo "Using electron data $DATA_ELECTRONS" >> $LOGFILE
echo "Using kaon data $DATA_KAONS" >> $LOGFILE
echo "Using photon data $DATA_PHOTONS" >> $LOGFILE
echo "Using pion data $DATA_PIONS" >> $LOGFILE
echo "Using output directory $OUTPUT_DIR" >> $LOGFILE
echo "Using $N_FILES files per particle type" >> $LOGFILE
echo "" >> $LOGFILE

echo "Starting prediction for electrons" >> $LOGFILE
predict_hgcal.py $MODEL $DATA_ELECTRONS $OUTPUT_ELECTRONS --max_files $N_FILES
echo "Starting prediction for kaons" >> $LOGFILE
predict_hgcal.py $MODEL $DATA_KAONS $OUTPUT_KAONS --max_files $N_FILES
echo "Starting prediction for photons" >> $LOGFILE
predict_hgcal.py $MODEL $DATA_PHOTONS $OUTPUT_PHOTONS --max_files $N_FILES
echo "Starting prediction for pions" >> $LOGFILE
predict_hgcal.py $MODEL $DATA_PIONS $OUTPUT_PIONS --max_files $N_FILES

ANALYSIS_ELECTRONS=$OUTPUT_ELECTRONS/analysis.bin.gz
ANALYSIS_KAONS=$OUTPUT_KAONS/analysis.bin.gz
ANALYSIS_PHOTONS=$OUTPUT_PHOTONS/analysis.bin.gz
ANALYSIS_PIONS=$OUTPUT_PIONS/analysis.bin.gz

echo "Starting analysis of electrons" >> $LOGFILE
cd $OUTPUT_ELECTRONS
analyse_hgcal_predictions.py $OUTPUT_ELECTRONS --analysisoutpath $ANALYSIS_ELECTRONS --slim
echo "Starting analysis of kaons" >> $LOGFILE
cd $OUTPUT_KAONS
analyse_hgcal_predictions.py $OUTPUT_KAONS --analysisoutpath $ANALYSIS_KAONS --slim
echo "Starting analysis of photons" >> $LOGFILE
cd $OUTPUT_PHOTONS
analyse_hgcal_predictions.py $OUTPUT_PHOTONS --analysisoutpath $ANALYSIS_PHOTONS --slim
echo "Starting analysis of pions" >> $LOGFILE
cd $OUTPUT_PIONS
analyse_hgcal_predictions.py $OUTPUT_PIONS --analysisoutpath $ANALYSIS_PIONS --slim
cd

echo "Starting plotting of electrons" >> $LOGFILE
python3 $HGCALML/scripts/resolution.py $ANALYSIS_ELECTRONS $OUTPUT_DIR
echo "Starting plotting of kaons" >> $LOGFILE
python3 $HGCALML/scripts/resolution.py $ANALYSIS_KAONS $OUTPUT_DIR
echo "Starting plotting of photons" >> $LOGFILE
python3 $HGCALML/scripts/resolution.py $ANALYSIS_PHOTONS $OUTPUT_DIR
echo "Starting plotting of pions" >> $LOGFILE
python3 $HGCALML/scripts/resolution.py $ANALYSIS_PIONS $OUTPUT_DIR

echo "" >> $LOGFILE
echo "DONE" >> $LOGFILE
