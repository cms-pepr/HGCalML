from datastructures.TrainData_NanoML import TrainData_NanoML
import numpy as np

class TrainData_NanoML2(TrainData_NanoML):
    def __init__(self):
        super().__init__()

    def interpretAllModelInputs(self, ilist, returndict=True):
        assert returndict

        out = {
            'features':ilist[0],
            'rechit_energy': ilist[0][:,0:1],
            't_idx':ilist[2],
            't_energy':ilist[4],
            't_pos':ilist[6],
            't_time':ilist[8],
            't_pid':ilist[10],
            't_spectator':ilist[12],
            't_fully_contained':ilist[14],
            'row_splits':ilist[1],
            't_rec_energy':ilist[16],
            't_is_unique':ilist[18],
            't_only_minbias':ilist[20],
            't_shower_class':ilist[22],
            }
        return out

    def createTruthDict(self, allfeat, truthidx=None):
        '''
        This is deprecated and should be replaced by a more transparent way.
        '''
        print(__name__, 'createTruthDict: should be deprecated soon and replaced by a more uniform interface')
        out=super().createTruthDict(allfeat)
        data = self.interpretAllModelInputs(allfeat, returndict=True)
        out['t_only_minbias'] = data['t_only_minbias']
        out['t_shower_class'] = data['t_shower_class']
        print("Calling NanoML2", np.unique(data['t_shower_class'], return_counts=True))
        return out

