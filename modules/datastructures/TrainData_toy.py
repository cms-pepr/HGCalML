



from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy 

class TrainData_toy(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="tree" #input root tree name
        
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
        
        
        #self.registerBranches(['']) #list of branches to be used 
        
        #self.registerBranches(self.truthclasses)
        
        self.remove=False
        self.weight=False
        
        #call this at the end
        self.reduceTruth(None)
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        from toygenerator import create_images
        
        feature_array, trutharray = create_images(3000,npixel=64)
        self.nsamples = len(feature_array)
        self.x=[feature_array] 
        self.y=[trutharray] # we need the features also in the truth part for weighting
        self.w=[] # no event weights


