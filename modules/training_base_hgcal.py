#from DeepJetCore.training.training_base import training_base


import concurrent.futures
import numpy as np

## to call it from cammand lines
import sys
import os
from argparse import ArgumentParser
import shutil
from DeepJetCore import DataCollection
import tensorflow.keras as keras
import tensorflow as tf
import copy
from DeepJetCore.training.gpuTools import DJCSetGPUs
from DeepJetCore.training.training_base import training_base as training_base_djc
import time



#for multi-gpu we need to overwrite a few things here

###
#
# this will become a cleaned-up version of DJC training_base at some point
#
###
class training_base(object):
    
    def __init__(
				self, splittrainandtest=0.85,
				useweights=False, testrun=False,
                testrun_fraction=0.1,
				resumeSilently=False, 
				renewtokens=False, #compat
				collection_class=DataCollection,
				parser=None,
                recreate_silently=False
				):
        
        scriptname=sys.argv[0]
        
        if parser is None: parser = ArgumentParser('Run the training')
        parser.add_argument('inputDataCollection')
        parser.add_argument('outputDir')
        #parser.add_argument('--modelMethod', help='Method to be used to instantiate model in derived training class', metavar='OPT', default=None)
        parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default="")
        #parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)
        #parser.add_argument("--submitbatch",  help="submits the job to condor" , default=False, action="store_true")
        #parser.add_argument("--walltime",  help="sets the wall time for the batch job, format: 1d5h or 2d or 3h etc" , default='1d')
        #parser.add_argument("--isbatchrun",   help="is batch run", default=False, action="store_true")
        parser.add_argument("--valdata",   help="set validation dataset (optional)", default="")
        parser.add_argument("--takeweights",   help="Applies weights from the model given as relative or absolute path. Matches by names and skips layers that don't match.", default="")
        
        
        args = parser.parse_args()
        self.args = args
        self.argstring = sys.argv
        #sanity check
        
        
        import matplotlib
        #if no X11 use below
        matplotlib.use('Agg')
        DJCSetGPUs(args.gpu)
        
        
        self.ngpus=1
        
        if len(args.gpu):
            self.ngpus=len([i for i in args.gpu.split(',')])
            print('running on '+str(self.ngpus)+ ' gpus')
            
        self.keras_inputs=[]
        self.keras_inputsshapes=[]
        self.keras_model=None
        self.mgpu_keras_models = []
        self.keras_weight_model_path=args.takeweights
        self.train_data=None
        self.val_data=None
        self.startlearningrate=None
        self.optimizer=None
        self.trainedepoches=0
        self.compiled=False
        self.checkpointcounter=0
        self.callbacks=None
        self.custom_optimizer=False
        self.copied_script=""
        
        self.inputData = os.path.abspath(args.inputDataCollection) \
												 if ',' not in args.inputDataCollection else \
														[os.path.abspath(i) for i in args.inputDataCollection.split(',')]
        self.outputDir=args.outputDir
        # create output dir
        
        isNewTraining=True
        if os.path.isdir(self.outputDir):
            if not (resumeSilently or recreate_silently):
                var = input('output dir exists. To recover a training, please type "yes"\n')
                if not var == 'yes':
                    raise Exception('output directory must not exists yet')
            isNewTraining=False
            if recreate_silently:
                isNewTraining=True     
        else:
            os.mkdir(self.outputDir)
        self.outputDir = os.path.abspath(self.outputDir)
        self.outputDir+='/'
        
        if recreate_silently:
            os.system('rm -rf '+ self.outputDir +'*')
        
        #copy configuration to output dir
        try:
            shutil.copyfile(scriptname,self.outputDir+os.path.basename(scriptname))
        except shutil.SameFileError:
            pass
        except BaseException as e:
            raise e
            
        self.copied_script = self.outputDir+os.path.basename(scriptname)
        
        self.train_data = collection_class()
        self.train_data.readFromFile(self.inputData)
        self.train_data.useweights=useweights
        
        if len(args.valdata):
            print('using validation data from ',args.valdata)
            self.val_data = DataCollection(args.valdata)
        
        else:
            if testrun:
                if len(self.train_data)>1:
                    self.train_data.split(testrun_fraction)
            
                self.train_data.dataclass_instance=None #can't be pickled
                self.val_data=copy.deepcopy(self.train_data)
                
            else:    
                self.val_data=self.train_data.split(splittrainandtest)
        


        shapes = self.train_data.getNumpyFeatureShapes()
        inputdtypes = self.train_data.getNumpyFeatureDTypes()
        inputnames= self.train_data.getNumpyFeatureArrayNames()
        for i in range(len(inputnames)): #in case they are not named
            if inputnames[i]=="" or inputnames[i]=="_rowsplits":
                inputnames[i]="input_"+str(i)+inputnames[i]


        print("shapes", shapes)
        print("inputdtypes", inputdtypes)
        print("inputnames", inputnames)
        
        self.keras_inputs=[]
        self.keras_inputsshapes=[]

        for s,dt,n in zip(shapes,inputdtypes,inputnames):
            self.keras_inputs.append(keras.layers.Input(shape=s, dtype=dt, name=n))
            self.keras_inputsshapes.append(s)
            
        #bookkeeping
        self.train_data.writeToFile(self.outputDir+'trainsamples.djcdc',abspath=True)
        self.val_data.writeToFile(self.outputDir+'valsamples.djcdc',abspath=True)
            
        if not isNewTraining:
            kfile = self.outputDir+'/KERAS_check_model_last.h5'
            if not os.path.isfile(kfile):
                kfile = self.outputDir+'/KERAS_check_model_last' #savedmodel format
                if not os.path.isdir(kfile):
                    kfile=''
            if len(kfile):
                print('loading model',kfile)
                self.loadModel(kfile)
                self.trainedepoches=0
                if os.path.isfile(self.outputDir+'losses.log'):
                    for line in open(self.outputDir+'losses.log'):
                        valloss = line.split(' ')[1][:-1]
                        if not valloss == "None":
                            self.trainedepoches+=1
                else:
                    print('incomplete epochs, starting from the beginning but with pretrained model')
            else:
                print('no model found in existing output dir, starting training from scratch')
        
    def modelSet(self):
        return (not self.keras_model==None) and not len(self.keras_weight_model_path)
        
    def syncModelWeights(self):
        if len(self.mgpu_keras_models) < 2:
            return
        weights = self.mgpu_keras_models[0].get_weights()
        for model in self.mgpu_keras_models[1:]:
            model.set_weights(weights)
        
    def setModel(self,model,**modelargs):
        if len(self.keras_inputs)<1:
            raise Exception('setup data first') 
        
        with tf.device('/GPU:0'):
            self.keras_model=model(self.keras_inputs,**modelargs)
        
        if len(self.keras_weight_model_path):
            from DeepJetCore.modeltools import apply_weights_where_possible, load_model
            self.keras_model = apply_weights_where_possible(self.keras_model, 
                                         load_model(self.keras_weight_model_path))
        if not self.keras_model:
            raise Exception('Setting model not successful') 

        self.mgpu_keras_models = [self.keras_model] #zero model
        if self.ngpus > 1:
            for i in range(self.ngpus-1):
                with tf.device(f'/GPU:{i+1}'):
                    self.mgpu_keras_models.append(model(self.keras_inputs,**modelargs))
            #sync initial or loaded weights
            self.syncModelWeights()

    def saveCheckPoint(self,addstring=''):
        self.checkpointcounter=self.checkpointcounter+1 
        self.saveModel("KERAS_model_checkpoint_"+str(self.checkpointcounter)+"_"+addstring)    
           
    def _loadModel(self,filename):
        from tensorflow.keras.models import load_model
        keras_model=load_model(filename, custom_objects=custom_objects_list)
        optimizer=keras_model.optimizer
        return keras_model, optimizer
                
    def loadModel(self,filename):
        self.keras_model, self.optimizer = self._loadModel(filename)
        self.compiled=True
        
    def setCustomOptimizer(self,optimizer):
        self.optimizer = optimizer
        self.custom_optimizer=True
        
    def compileModel(self,
                     learningrate,
                     clipnorm=None,
                     print_models=False,
                     metrics=None,
                     is_eager=False,
                     **compileargs):

        if not self.keras_model:
            raise Exception('set model first') 

        print('Model being compiled for '+str(self.ngpus)+' gpus')
            
        self.startlearningrate=learningrate
        
        if not self.custom_optimizer:
            from tensorflow.keras.optimizers import Adam
            if clipnorm:
                self.optimizer = Adam(lr=self.startlearningrate,clipnorm=clipnorm)
            else:
                self.optimizer = Adam(lr=self.startlearningrate)
            
        def run_compile(model, device):
            with tf.device(device):
                model.compile(optimizer=self.optimizer,metrics=metrics,**compileargs)
                if is_eager:
                    #call on one batch to fully build it
                    print('Model being called once for device '+str(device))
                    model(self.train_data.getExampleFeatureBatch())
                    
        for i, m in enumerate(self.mgpu_keras_models):
            run_compile(m, f'/GPU:{i}')

        from DebugLayers import switch_off_debug_plots
        #run debug layers etc just for one model
        for i, m in enumerate(self.mgpu_keras_models):
            if i:
                switch_off_debug_plots(m)
        
        if print_models:
            print(self.mgpu_keras_models[0].summary())
        self.compiled=True

        
    def saveModel(self,outfile):
        self.keras_model.save(self.outputDir+outfile)
        

    # add some of the multi-gpu initialisation here?
    def _initTraining(self,
                      nepochs,
                     batchsize,
                     use_sum_of_squares=False):
        
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        self.train_data.batch_uses_sum_of_squares=use_sum_of_squares
        self.val_data.batch_uses_sum_of_squares=use_sum_of_squares
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)

        ## create multi-gpu models

    #now this is hgcal specific because of missing truth, think later how to do that better
    def compute_gradients(self, model, data, i):
        with tf.device(f'/GPU:{i}'):
            with tf.GradientTape() as tape:
                predictions = model(data, training=True)
                loss = tf.add_n(model.losses)
            return tape.gradient(loss, model.trainable_variables)
        
    def average_gradients(self, all_gradients):
        # Average the gradients across GPUs
        if len(all_gradients) < 2:
            return all_gradients[0]
        avg_grads = []
        for grad_list_tuple in zip(*all_gradients):
            grads = [g for g in grad_list_tuple if g is not None]
            avg_grads.append(tf.reduce_mean(grads, axis=0))
        return avg_grads

    def trainstep_parallel(self, split_data):

        batch_gradients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.ngpus) as executor:
            futures = [executor.submit(self.compute_gradients, self.mgpu_keras_models[i], split_data[i], i) for i in range(self.ngpus)]
            for future in concurrent.futures.as_completed(futures):
                gradients = future.result()
                batch_gradients.append(gradients)

        # Average gradients across GPUs - just see if it executes for now
        avg_grads = self.average_gradients(batch_gradients)
        self.optimizer.apply_gradients(zip(avg_grads, self.mgpu_keras_models[0].trainable_variables))
        self.syncModelWeights() # weights synced


        
    def trainModel(self,
                   nepochs,
                   batchsize,
                   run_eagerly=False,
                   batchsize_use_sum_of_squares = False,
                   fake_truth=False,#extend the truth list with dummies. Useful when adding more prediction outputs than truth inputs
                   stop_patience=-1, 
                   lr_factor=0.5,
                   lr_patience=-1, 
                   lr_epsilon=0.003, 
                   lr_cooldown=6, 
                   lr_minimum=0.000001,
                   checkperiod=10,
                   backup_after_batches=-1,
                   additional_plots=None,
                   additional_callbacks=None,
                   load_in_mem = False,
                   max_files = -1,
                   plot_batch_loss = False,
                   verbose = 0,
                   **trainargs):
        
        for m in self.mgpu_keras_models:
            m.run_eagerly=run_eagerly
        # write only after the output classes have been added
        self._initTraining(nepochs,batchsize, batchsize_use_sum_of_squares)
        
        try: #won't work for purely eager models
            self.keras_model.save(self.outputDir+'KERAS_untrained_model')
        except:
            pass
        print('setting up callbacks')
        from DeepJetCore.training.DeepJet_callbacks import DeepJet_callbacks
        
        self.callbacks=DeepJet_callbacks(self.keras_model,
                                    stop_patience=stop_patience, 
                                    lr_factor=lr_factor,
                                    lr_patience=lr_patience, 
                                    lr_epsilon=lr_epsilon, 
                                    lr_cooldown=lr_cooldown, 
                                    lr_minimum=lr_minimum,
                                    outputDir=self.outputDir,
                                    checkperiod=checkperiod,
                                    backup_after_batches=backup_after_batches,
                                    checkperiodoffset=self.trainedepoches,
                                    additional_plots=additional_plots,
                                    batch_loss = plot_batch_loss,
                                    print_summary_after_first_batch=run_eagerly,
                                    minTokenLifetime = -1)
        
        if additional_callbacks is not None:
            if not isinstance(additional_callbacks, list):
                additional_callbacks=[additional_callbacks]
            self.callbacks.callbacks.extend(additional_callbacks)
            
        #create callbacks wrapper
        
        
        
        #prepare generator 
    
        print("setting up generator... can take a while")
        use_fake_truth=None
        if fake_truth:
            if isinstance(self.keras_model.output,dict):
                use_fake_truth = [k for k in self.keras_model.output.keys()]
            elif isinstance(self.keras_model.output,list):
                use_fake_truth = len(self.keras_model.output)
                
        traingen = self.train_data.invokeGenerator(fake_truth = use_fake_truth)
        valgen = self.val_data.invokeGenerator(fake_truth = use_fake_truth)

        batch_time = 100


        

        while(self.trainedepoches < nepochs):
            
        
            #this can change from epoch to epoch
            #calculate steps for this epoch
            #feed info below
            traingen.prepareNextEpoch()
            valgen.prepareNextEpoch()
            nbatches_train = traingen.getNBatches() #might have changed due to shuffeling
            nbatches_val = valgen.getNBatches()
        
            print('>>>> epoch', self.trainedepoches,"/",nepochs)
            print('training batches: ',nbatches_train)
            print('validation batches: ',nbatches_val)


            #this is in here because it needs steps count
            callbacks = tf.keras.callbacks.CallbackList(
                    self.callbacks.callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self.keras_model, #only run them on the main module!
                    verbose=verbose,
                    epochs=1,
                    steps=nbatches_train,
                )
            if self.trainedepoches == 0:
                callbacks.on_train_begin()

            callbacks.on_epoch_begin(self.trainedepoches)

            ### 

            nbatches_in = 0
            single_counter = 0
            time_sum = 0

            while nbatches_in < nbatches_train:

                if batch_time > 0.1:
                    st = time.time()

                thisbatch = []
                while len(thisbatch) < self.ngpus and nbatches_in < nbatches_train:
                    #only 'feature' part matters for HGCAL
                    thisbatch.append(next(traingen.feedNumpyData())[0])
                    nbatches_in += 1

                if len(thisbatch) != self.ngpus: #last batch might not be enough
                    break

                callbacks.on_train_batch_begin(single_counter)

                self.trainstep_parallel(thisbatch)
                
                if batch_time > 0.1:
                    batch_time = time.time() - st
                    time_sum += batch_time
                    if not nbatches_in % (int(20/(time_sum/nbatches_in))+1): # print less when it's faster
                        print('avg batch time', time_sum/nbatches_in, 's')

                logs = { m.name: m.result() for m in self.keras_model.metrics } #only for main model

                callbacks.on_train_batch_end(single_counter, logs)

                single_counter += 1
            try:
                callbacks.on_epoch_end(self.trainedepoches, logs) #use same logs here
            except Exception as e:
                print(e)
                print('will continue training anyway')
            
            self.trainedepoches += 1
            traingen.shuffleFileList()
            #
    
        self.saveModel("KERAS_model.h5")

        return self.keras_model, callbacks.history
    
    
    
       
    def change_learning_rate(self, new_lr):
        import tensorflow.keras.backend as K
        K.set_value(self.keras_model.optimizer.lr, new_lr)
        
      

class HGCalTraining(training_base):
    def __init__(self, *args, 
                 **kwargs):
        '''
        Adds file logging
        '''
        #no reason for a lot of validation samples usually
        super().__init__(*args, resumeSilently=True,splittrainandtest=0.95,**kwargs)
        
        from config_saver import copyModules
        copyModules(self.outputDir)#save the modules with indexing for overwrites

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



class HGCalTraining_compat(training_base_djc):
    def __init__(self, *args, 
                 **kwargs):
        '''
        Adds file logging
        '''
        #no reason for a lot of validation samples usually
        super().__init__(*args, resumeSilently=True,splittrainandtest=0.95,**kwargs)
        
        from config_saver import copyModules
        copyModules(self.outputDir)#save the modules with indexing for overwrites

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