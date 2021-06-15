from DeepJetCore.training.training_base import *

import tensorflow.python.keras.engine.functional

class HGCalTrainingBase(training_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def loadModel(self,filename):
        super().loadModel(filename)
        with open(self.outputDir + 'model_type.txt', 'r') as f:
            text = str(f.read())
            if 'RobustModel' in text:
                self.keras_model = RobustModel(inputs=self.keras_model.inputs, outputs=self.keras_model.outputs)

    def setModel(self,model,**modelargs):
        super().setModel(model, **modelargs)

        with open(self.outputDir + 'model_type.txt', 'w') as f:
            f.write(str(type(self.keras_model)))

