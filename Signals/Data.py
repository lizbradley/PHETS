import numpy as np

class BaseTrajectory:

    def __init__(
            self, data,
            fname=None,
            crop=None,
            num_windows=1,
            window_length=None,
            normalize_volume=None,      # 'window', 'crop', 'full' or None
            time_units='seconds'        # 'seconds' or 'samples'


    ):
        if isinstance(data, basestring):        # is filename
            self.data_full = np.loadtxt(data)
            self.fname = data
        else:                                   # is array
            self.data_full = data
            self.fname = fname


        self.time_units = time_units

        self.crop = crop
        self.data = self.apply_crop()

        self.num_windows=num_windows,
        self.window_length=window_length

        self.norm_vol = normalize_volume

    def apply_crop(self, crop):
        if self.crop is None:
            return self.data_full
        else:
            # do crop
            pass

    def slice(self):
        pass

    def apply_vol_norm(self):
        pass




class TimeSeries(BaseTrajectory):

    def __init__(self, data, fname=None):
        BaseTrajectory.__init__(self, data, fname)







