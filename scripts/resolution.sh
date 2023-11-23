#!/bin/bash

MODEL="/mnt/home/pzehetner/Connecting-The-Dots/include_noisefilterPUContinue/KERAS_check_best_model.h5"
MODEL_ID="OLD"
BASEPATH="/mnt/ceph/users/pzehetner/Paper/Test/granular_pu_tests/pu_test_events"
PREDDIR="/mnt/ceph/users/pzehetner/Paper/predictions/"
PARTICLES=( "electrons" "photons" "kaons" "pions")
ENERGIES=( "10" "20" "50" "75" "100" "125" "150" "175" "200" "250" )
PU=( "0" "40" "200" )
COUNTER=0
OUTPUTFILE=analysis.bin.gz

# print particles

for particle in ${PARTICLES[@]}
do
    for energy in ${ENERGIES[@]}
    do
        for pu in ${PU[@]}
        do
            echo "particle: $particle, energy: $energy, pu: $pu"
            directory="$BASEPATH/${particle}_${energy}GeV_eta20_PU${pu}"
            outputdir="${PREDDIR}/${MODEL_ID}_${particle}_${energy}GeV_eta20_PU${pu}"
	    echo "Data: ${directory}"
	    echo "Output: ${outputdir}"

            # if counter > 10: break
            if [ $COUNTER -gt 0 ]; then
                break
            fi

            # Skip if directory does not exist
            if [ ! -d $directory ]; then
                echo "Skipping $directory"
                continue
            fi

            # Check for dataCollection
            datacollection="$directory/dataCollection.djcdc"
            if [ ! -f $datacollection ]; then
		echo "Create DataCollection"
		cd $directory
		createDataCollectionFromTD.py -c TrainData_NanoML -o dataCollection.djcdc *.djctd
		cd -
		echo ""
		echo ""
		echo ""
            fi

            # Skip if output file already exists
            if [ -f $directory/$OUTPUTFILE ]; then
                echo "Skipping $directory already processed"
                continue
            fi

            if [ -f $directory/pred.flag ]; then
                echo "Other process already working on this"
                continue
            fi

            if [ ! -f ${outputdir}/pred_00000.bin.gz ]; then
		echo "Running prediction"
                touch $directory/pred.flag
                predict_hgcal.py $MODEL $datacollection $outputdir
                rm $directory/pred.flag
            fi

            if [ -f $outputdir/analysis.flag ]; then
                echo "Other process already working on this"
                continue
            fi

            PWD=`pwd`
            cd $outputdir
            if [ ! -f $OUTPUTFILE ]; then
                touch analysis.flag
                analyse_hgcal_predictions.py . --analysisoutpath $OUTPUTFILE --slim -b 0.3 -d 0.25 -i 0.333 --picturepath full_classic_d025_i033
                rm analysis.flag

                COUNTER=$((COUNTER+1))
                echo $COUNTER
            fi
        done
    done
done
