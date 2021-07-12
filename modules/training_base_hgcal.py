from DeepJetCore.training.training_base import *

class HGCalTraining(training_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def loadModel(self,filename):
        if str(filename).endswith('.h5'):
            filename = os.path.splitext(filename)[0] + '_save'

        if os.path.exists(filename):
            self.keras_model = keras.models.load_model(filename, custom_objects=custom_objects_list)
        else:
            super().loadModel(filename)

