import numpy as np
from matplotlib import scale as mscale
import matplotlib
from matplotlib.ticker import FixedLocator, FuncFormatter


class ResponseScale(mscale.ScaleBase):
    name='response_scale'

    def __init__(self, axis):
        super().__init__(axis)

    def get_transform(self):
        return self.ResponseTransform()

    class ResponseTransform(matplotlib.scale.FuncTransform):
        def __init__(self):
            super().__init__(forward=self.forward, inverse=self.inverse)

        def forward(self, array):
            return np.where(np.less_equal(array,3), array, 3+np.log(array)-np.log(3))

        def inverse(self, array):
            return np.where(np.less_equal(array,3), array, np.exp(array+np.log(3) - 3))

    def set_default_locators_and_formatters(self, axis):
        fmt = FuncFormatter(
            lambda x, pos=None: str(x))

        locaters = [0,0.5,1,1.5,2,2.5,3,4,5,10,50,100,1000,10000,10000,100000,1000000,10000000,10000000,10000000]

        axis.set(major_locator=FixedLocator(locaters),
                 major_formatter=fmt, minor_formatter=fmt)


def register():
    mscale.register_scale(ResponseScale)

