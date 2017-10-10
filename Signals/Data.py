import numpy as np

class BaseTrajectory:

    def __init__(
            self, data,
            fname=None,
            crop=None,
            num_windows=1,
            window_length=None,
            normalize_volume=None,      # 'window', 'crop', 'full' or None


    ):
        if isinstance(data, basestring):        # is filename
            self.data_full = np.loadtxt(data)
            self.fname = data
        else:                                   # is array
            self.data_full = data
            self.fname = fname


        self.norm_vol = normalize_volume
        self.crop_lim = crop
        self.num_windows=num_windows,
        self.window_length=window_length

        self.data = self.crop()
        self.windows, self.start_pts = self.slice()

    def crop(self):
        if self.crop_lim is None:
            return self.data_full
        else:
            return self.data_full[self.crop_lim[0]:self.crop_lim[1]]



    def slice(self):
        # use np.slice_array
        # start pts?

        if self.norm_vol[2]:
            windows = [np.true_divide(w, np.max(np.abs(w))) for w in windows]

        return windows, start_pts





class TimeSeries(BaseTrajectory):

    def __init__(self, data, fname=None):
        BaseTrajectory.__init__(self, data, fname)






class Trajectory(BaseTrajectory):

    def __init__(self, data, fname=None):
        BaseTrajectory.__init__(self, data, fname)
