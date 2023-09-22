

export HGCALML=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALML

export PATH=$HGCALML/scripts:$PATH
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
export LD_LIBRARY_PATH=$HGCALML/modules:$LD_LIBRARY_PATH
export LC_ALL=C.UTF-8 	# necessary for wandb
export LANG=C.UTF-8	# necessary for wandb
#?export PYTHONPATH=$HGCALML/modules/datastructures:$PYTHONPATH
#for ffmpeg

# wandb api key
# if ~/private/wandb_api.sh file exists then source from it
if [[ -f ~/private/wandb_api.sh ]]; then
   source ~/private/wandb_api.sh
fi

