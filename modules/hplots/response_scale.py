import numpy as np
from matplotlib import scale as mscale
import matplotlib
from matplotlib.ticker import FixedLocator, FuncFormatter, Locator, AutoLocator


class ResponseLocator(Locator):
    def __init__(self, locs, nbins=None,numticks=None):
        self.locs = np.asarray(locs)
        self.numticks=numticks
        self.auto_locator = AutoLocator()
        self.nbins=nbins

    def set_params(self, nbins=None,numticks=None):
        """Set parameters within this locator."""
        if nbins is not None:
            self.nbins = nbins
        if numticks is not None:
            self.numticks=numticks

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    @property
    def numticks(self):
        # Old hard-coded default.
        return self._numticks if self._numticks is not None else 11

    @numticks.setter
    def numticks(self, numticks):
        self._numticks = numticks

    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """
        if vmax < 3:
            return np.round(self.auto_locator.tick_values(vmin, vmax),4)
        else:
            if self.nbins is None:
                return self.locs
            step = max(int(np.ceil(len(self.locs) / self.nbins)), 1)
            ticks = self.locs[::step]
            for i in range(1, step):
                ticks1 = self.locs[i::step]
                if np.abs(ticks1).min() < np.abs(ticks).min():
                    ticks = ticks1
            return self.raise_if_exceeds(ticks)


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

        axis.set(major_locator=ResponseLocator(locaters),
                 major_formatter=fmt, minor_formatter=fmt)


def register():
    mscale.register_scale(ResponseScale)

