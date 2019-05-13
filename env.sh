
#! /bin/bash
THISDIR=`pwd`
cd /afs/cern.ch/user/j/jkiesele/work/TESTDJ/DeepJetCore
source env.sh
cd $THISDIR
export HGCALML=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALML

cd $HGCALML
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
export PYTHONPATH=$HGCALML/modules/datastructures:$PYTHONPATH
