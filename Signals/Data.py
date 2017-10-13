import numpy as np


class BaseTrajectory:

	def __init__(
			self, data,
			fname=None,
			crop=None,
			num_windows=None,
			window_length=None,
			vol_norm=(False, False, False),     # (full, crop, windows)


	):
		if isinstance(data, basestring):        # is filename
			self.data_full = np.loadtxt(data)
			self.fname = data
		else:                                   # is array
			self.data_full = data
			self.fname = fname


		self.norm_vol = vol_norm
		self.crop_lim = crop
		self.num_windows = num_windows
		self.window_length = window_length

		if self.norm_vol[0]:
			self.data_full = self.normalize(self.data_full)
		self.data = self.crop(crop)
		self.windows, self.win_start_idxs = self.slice(
			 num_windows, window_length
		)

	@staticmethod
	def normalize(data):
		return np.true_divide(data, np.max(data))

	def crop(self, lim):
		if lim is None:
			data = self.data_full
		else:
			data = self.data_full[self.crop_lim[0]:self.crop_lim[1]]

		if self.norm_vol[1]:
			data = self.normalize(data)
			self.data = data
		return data



	def slice(self, num_windows, window_length):
		if num_windows is None:
			return None, None
		else:
			start_idxs = np.floor(
				np.linspace(0, len(self.data), num_windows)
			).astype(int)

			if window_length is None:
				window_length = len(self.data) / num_windows
			windows = [self.data[sp:sp + window_length] for sp in start_idxs]

		if self.norm_vol[2]:
			windows = [self.normalize(w) for w in windows]
			self.windows, self.win_start_idxs = windows, start_idxs
		return windows, start_idxs



class TimeSeries(BaseTrajectory):

	def __init__(self, data, fname=None):
		BaseTrajectory.__init__(self, data, fname)






class Trajectory(BaseTrajectory):

	def __init__(self, data, fname=None):
		BaseTrajectory.__init__(self, data, fname)
