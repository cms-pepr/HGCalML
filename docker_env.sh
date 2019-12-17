

export HGCALML=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALML

export PATH=$HGCALML/scripts:$PATH
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
#?export PYTHONPATH=$HGCALML/modules/datastructures:$PYTHONPATH
#for ffmpeg
export PATH=$PATH:/afs/cern.ch/work/j/jkiesele/public/conda_env/miniconda3/envs/djcenv/bin/
