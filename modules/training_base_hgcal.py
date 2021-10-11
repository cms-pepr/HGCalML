from DeepJetCore.training.training_base import *

class HGCalTraining(training_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_data.writeToFile(self.outputDir + 'valsamples.djcdc')

    
    def loadModel(self,filename):
        if str(filename).endswith('.h5'):
            newfilename = os.path.splitext(filename)[0] + '_save'
            if os.path.exists(filename):
                super().loadModel(newfilename)
                return
            
        super().loadModel(filename)


    
    def setModel(self,model,**modelargs):
        if len(self.keras_inputs)<1:
            raise Exception('setup data first') 
        if self.dist_strat_scope is not None:
            with self.dist_strat_scope.scope():
                self.keras_model=model(self.keras_inputs,**modelargs)
        else:
            self.keras_model=model(self.keras_inputs,**modelargs)
        if hasattr(self.keras_model, "_is_djc_keras_model"): #compatibility
            self.keras_model.setInputShape(self.keras_inputs)
            self.keras_model.build(None)
            
        if len(self.keras_weight_model_path):
            from DeepJetCore.modeltools import apply_weights_where_possible
            
            newfilename=self.keras_weight_model_path
            if str(newfilename).endswith('.h5'):
                newfilename = os.path.splitext(newfilename)[0] + '_save'
            
            weightmodel = tf.keras.models.load_model(newfilename, custom_objects=custom_objects_list)
            if isinstance(self.keras_model, (RobustModel,ExtendedMetricsModel)):
                self.keras_model.model = apply_weights_where_possible(self.keras_model.model, weightmodel.model)
            else:
                self.keras_model = apply_weights_where_possible(self.keras_model, weightmodel)
        #try:
        #    self.keras_model=model(self.keras_inputs,**modelargs)
        #except BaseException as e:
        #    print('problem in setting model. Reminder: since DJC 2.0, NClassificationTargets and RegressionTargets must not be specified anymore')
        #    raise e
        if not self.keras_model:
            raise Exception('Setting model not successful') 
        
    