#!/bin/bash

# Script to run the analysis multiple times and compare the results


# Parameters to scan

# distance threshold    d: 0.3, 0.5, 0.7
# beta threshold        b: 0.1, 0.3, 0.5
# iou threshold         i: 0.1, 0.2, 0.5

PREDICTION_DIR=$1
OUTPUT_BASE=$2
CURRENT_DIR=`pwd`

if [ -d "$OUTPUT_BASE" ]; then
    echo "Output base folder already exists, exiting"
    exit 1
else
    mkdir $OUTPUT_BASE
fi

LOGFILE=$OUTPUT_BASE/log.txt
echo "Logging to $LOGFILE"
echo "Logging to $LOGFILE" > $LOGFILE
echo "Using predictions from $PREDICTION_DIR" >> $LOGFILE
echo "Storing results in $OUTPUT_BASE" >> $LOGFILE
echo "Testing parameters b=0.1,0.3,0.5" >> $LOGFILE
echo "Testing parameters d=0.3,0.5,0.7" >> $LOGFILE
echo "Testing parameters i=0.1,0.2,0.5" >> $LOGFILE


# for b in 0.1
for b in 0.1 0.3 0.5 0.6
do
    # for d in 0.3
    for d in 0.2 0.3 0.4 0.5
    do
        for i in 0.1 0.2
        # for i in 0.1
        do
            directory=${OUTPUT_BASE}/b${b}_d${d}_i${i}
            echo "Running analysis with b=$b, d=$d, i=$i" >> $LOGFILE
	    echo "Saving to directory $directory" >> $LOGFILE
            mkdir $directory
            analyse_hgcal_predictions.py $PREDICTION_DIR -b $b -d $d -i $i \
                --analysisoutpath $directory/analysis.bin.gz --picturepath $directory --slim
        done
        echo "" >> $LOGFILE
    done
    echo "" >> $LOGFILE
done
