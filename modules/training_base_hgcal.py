from DeepJetCore.training.training_base import training_base
from argparse import ArgumentParser
import tensorflow as tf


def unpack_ragged(inputs):
    output = []
    for i in inputs:
        output += [i.values, i.row_splits]
    return output
        
class HGCalTraining(training_base):
    def __init__(self, *args, 
                 parser = None,
                 **kwargs):
        '''
        Adds file logging
        '''
        #use the DJC training base option to pass a parser
        if parser is None:
            parser = ArgumentParser('Run the training')
        parser.add_argument("--interactive",   help="prints output to screen", default=False, action="store_true")
        
        #no reason for a lot of validation samples usually
        super().__init__(*args, resumeSilently=True,parser=parser,splittrainandtest=0.95,**kwargs)
        
        if not self.args.interactive:
            print('>>> redirecting the following stdout and stderr to logs in',self.outputDir)
            import sys
            sys.stdout = open(self.outputDir+'/stdout.txt', 'w')
            sys.stderr = open(self.outputDir+'/stderr.txt', 'w')
            
        
        from config_saver import copyModules
        copyModules(self.outputDir)#save the modules with indexing for overwrites

    def pack_to_ragged_batch(self, data_list):
        feat_list = data_list[0] #only features
        truth_list = data_list[1]

        def pack_list(mylist):
            o=[]
            for i in range(len(mylist)//2):
                o.append(tf.RaggedTensor.from_row_splits(values=mylist[2*i], row_splits=mylist[2*i + 1][:,0]))
            return o
        fl = pack_list(feat_list)
        return ( fl , truth_list ) #pass truth list, this is anyway a dummy for hgcal

    def wrap_model_ragged(self, rin):
        fin = unpack_ragged(rin)
        return self.keras_model(fin)
        
    def compileModel(self, **kwargs):
        super().compileModel(is_eager=True,
                    loss=None,
                    **kwargs)
    
    def trainModel(self,
                   nepochs,
                   batchsize,
                   run_eagerly=True,
                   verbose=1,
                   batchsize_use_sum_of_squares = False,
                   fake_truth=True,#extend the truth list with dummies. Useful when adding more prediction outputs than truth inputs
                   backup_after_batches=500,
                   checkperiod=1,
                   stop_patience=-1, 
                   lr_factor=0.5,
                   lr_patience=-1, 
                   lr_epsilon=0.003, 
                   lr_cooldown=6, 
                   lr_minimum=0.000001,
                   additional_plots=None,
                   additional_callbacks=None,
                   load_in_mem = False,
                   max_files = -1,
                   plot_batch_loss = False,
                   **trainargs):
        
        self.keras_model.run_eagerly=run_eagerly
        # write only after the output classes have been added
        self._initTraining(nepochs,batchsize, batchsize_use_sum_of_squares)
        
        try: #won't work for purely eager models
            self.keras_model.save(self.outputDir+'KERAS_untrained_model')
        except:
            pass
        print('setting up callbacks')
        from DeepJetCore.training.DeepJet_callbacks import DeepJet_callbacks
        minTokenLifetime = 5
        if not self.renewtokens:
            minTokenLifetime = -1
        
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
                                    minTokenLifetime = minTokenLifetime)
        
        if additional_callbacks is not None:
            if not isinstance(additional_callbacks, list):
                additional_callbacks=[additional_callbacks]
            self.callbacks.callbacks.extend(additional_callbacks)
            
        
        print('starting training')
        if load_in_mem:
            print('make features')
            X_train = self.train_data.getAllFeatures(nfiles=max_files)
            X_test = self.val_data.getAllFeatures(nfiles=max_files)
            print('make truth')
            Y_train = self.train_data.getAllLabels(nfiles=max_files)
            Y_test = self.val_data.getAllLabels(nfiles=max_files)
            self.keras_model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepochs,
                                 callbacks=self.callbacks.callbacks,
                                 validation_data=(X_test, Y_test),
                                 max_queue_size=1,
                                 use_multiprocessing=False,
                                 workers=0,    
                                 **trainargs)
        else:
        
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

                #data = self.to_ragged_tensor(traingen.feedNumpyData())
                #data = traingen.feedNumpyData()
                
                for batch in traingen.feedNumpyData():
                    #here we would need to aggregate N_gpu batches before passing them on
                    
                    with tf.GradientTape() as tape:
                        rdata = self.pack_to_ragged_batch(batch) 
                        o = self.wrap_model_ragged(rdata[0])
                        
                        losses =  self.keras_model.losses
                        losses =  tf.add_n(losses) 

                    grads = tape.gradient( losses , self.keras_model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.keras_model.trainable_variables))

                #self.keras_model.fit(data, 
                #                     steps_per_epoch=nbatches_train,
                #                     epochs=self.trainedepoches + 1,
                #                     initial_epoch=self.trainedepoches,
                #                     callbacks=self.callbacks.callbacks,
                #                     validation_data=valgen.feedNumpyData(),
                #                     validation_steps=nbatches_val,
                #                     max_queue_size=1,
                #                     use_multiprocessing=False,
                #                     workers=0,
                #                     **trainargs
                #)
                self.trainedepoches += 1
                traingen.shuffleFileList()
                #
        
            self.saveModel("KERAS_model.h5")

        return self.keras_model, self.callbacks.history
    
