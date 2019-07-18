


from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy 

class TrainData_hitlist(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="Delphes" #input root tree name
        
        self.feat_branch="rechit_features"
        self.truth_branch="rechit_simcluster_fractions"
        # this needs to be adapted!!
        self.max_rechits = 3500
        
        #this should be fine
        self.n_features=10
        self.n_simcluster=20
        
        self.regressiontargetclasses = [str(i) for i in range(self.n_simcluster)]
        
        
        self.other_useless_inits()
        
        
    
    
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        
        import ROOT
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import  readListArray
        
        feature_array, n_rechits_cut = readListArray(filename,
                                      self.treename,
                                      self.feat_branch,
                                      self.nsamples,
                                      list_size=self.max_rechits, 
                                      n_feat_per_element=self.n_features,
                                      zeropad=True,
                                      list_size_cut=True)
        
        
        energy_only = feature_array[:,:,0:1]#keep dimension
        
        
        fraction_array,_ = readListArray(filename,
                                       self.treename,
                                       self.truth_branch,
                                       self.nsamples,
                                      list_size=self.max_rechits, 
                                      n_feat_per_element=self.n_simcluster,#nsimcluster, right now just one, but zero-padded here
                                      zeropad=True,
                                      list_size_cut=True)
        
        print('TrainData_hitlistX: ' ,filename,';convert from root: fraction of hits cut ', 100.*float(n_rechits_cut)/float(self.nsamples),'%')
        
        #needs the energy, too to determine weights
        fraction_array = numpy.concatenate([fraction_array,energy_only],axis=-1)
        #in case something was removed here
        if n_rechits_cut>0:
            feature_array  = feature_array[0:self.nsamples-n_rechits_cut]
            fraction_array = fraction_array[0:self.nsamples-n_rechits_cut]
        
        
        self.nsamples=len(feature_array)
        
        self.x=[feature_array] 
        self.y=[fraction_array] # we need the features also in the truth part for weighting
        self.w=[] # no event weights










    ### below, some standard values are set, but we don't need them here
    #just to speed up, we don't need weights
    def produceBinWeighter(self, filename):
        return self.make_empty_weighter()
    
    #not needed, override
    def produceMeansFromRootFile(self,orig_list, limit):
        import numpy
        return numpy.array([1.,])
    
    def other_useless_inits(self):
        self.truthclasses=[] #truth classes for classification
        self.weightbranchX='isA' #needs to be specified if weighter is used
        self.weightbranchY='isB' #needs to be specified if weighter is used
        #there is no need to resample/reweight
        self.weight=False
        self.remove=False
        #does not do anything in this configuration
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,40000],dtype=float) 
        self.weight_binY = numpy.array([0,40000],dtype=float) 
        #call this at the end
        self.reduceTruth(None)
        self.registerBranches(self.truthclasses)







class TrainData_hitlist_layercluster(TrainData_hitlist):
    def __init__(self):
        TrainData_hitlist.__init__(self)

        self.feat_branch="layercluster_features"
        self.truth_branch="layercluster_simcluster_fractions"
        # this needs to be adapted!!
        self.max_rechits = 1200
        #this should be fine
        self.n_features=17
        self.n_simcluster=20
        



