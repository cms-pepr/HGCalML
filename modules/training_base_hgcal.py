from DeepJetCore.training.training_base import training_base

class HGCalTraining(training_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def trainModel(self,
                   nepochs,
                   batchsize,
                   **kwargs):
        '''
        Just implements some defaults
        '''
        return super().trainModel(nepochs=nepochs,
                           batchsize=batchsize,
                           run_eagerly=True,
                           verbose=2,
                           batchsize_use_sum_of_squares=False,
                           fake_truth=True,
                           checkperiod=1, 
                           backup_after_batches=500,
                            **kwargs)