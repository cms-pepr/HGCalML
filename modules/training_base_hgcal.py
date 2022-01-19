from DeepJetCore.training.training_base import training_base

class HGCalTraining(training_base):
    def __init__(self, *args, 
                 redirect_stdout=True,
                 **kwargs):
        '''
        Adds file logging
        '''
        super().__init__(*args, resumeSilently=True,**kwargs)
        
        if redirect_stdout:
            print('>>> redirecting the following stdout and stderr to logs in',self.outputDir)
            import sys
            sys.stdout = open(self.outputDir+'/stdout.txt', 'w')
            sys.stderr = open(self.outputDir+'/stderr.txt', 'w')

    def compileModel(self, **kwargs):
        super().compileModel(is_eager=True,
                       loss=None,
                       **kwargs)
    
    def trainModel(self,
                   nepochs,
                   batchsize,
                   backup_after_batches=500,
                   checkperiod=1, 
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
                           backup_after_batches=backup_after_batches,
                           checkperiod=checkperiod,
                            **kwargs)