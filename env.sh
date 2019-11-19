
#! /bin/bash


#eval "$(/afs/cern.ch/work/j/jkiesele/public/conda_env/miniconda3/bin/conda shell.bash hook)"

THISDIR=`pwd`
cd /afs/cern.ch/work/j/jkiesele/public/conda_env/DeepJetCore
source env.sh
cd $THISDIR
export HGCALML=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALML


cd $HGCALML
export PATH=$HGCALML/scripts:$PATH
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
export PYTHONPATH=$HGCALML/modules/datastructures:$PYTHONPATH
