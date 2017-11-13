import sys
import numpy as np

import plots
from DCE import embed
from PH import Filtration
from config import SAMPLE_RATE
from utilities import print_title, print_still


class BaseTrajectory(object):

	def __init__(self,
	        data,
	        name=None,
	        fname=None,
	        skiprows=0,
			crop=None,
			num_windows=None,
			window_length=None,
			vol_norm=(False, False, False),     # (full, crop, windows)
	        time_units='samples'

	):
		if isinstance(data, basestring):        # is filename
			print 'loading input file...'
			self.data_full = np.loadtxt(data, skiprows=skiprows)
			self.fname = data
		else:                                   # is array
			self.data_full = data
			self.fname = fname

		if name is not None:
			self.name = name
		elif self.fname is not None:
			self.name = self.fname.split('/')[-1].split('.')[0]
		else:
			self.name = None

		self.norm_vol = vol_norm
		self.crop_lim = crop
		self.num_windows = num_windows
		self.window_length = window_length
		self.time_units = time_units

		if self.norm_vol[0]:
			self.data_full = self.normalize(self.data_full)

		self.data = None
		self.windows = None
		self.win_start_pts = None
		self.crop(crop)
		self.slice(num_windows, window_length)


	@staticmethod
	def normalize(data):
		return np.true_divide(data, np.max(np.abs(data)))


	def crop(self, lim):
		if lim is None:
			self.data = self.data_full
			self.crop_lim = (0, len(self.data))

		else:
			crop_lim = np.array(self.crop_lim)
			self.crop_lim = crop_lim

			if self.time_units == 'seconds':
				crop_lim = (crop_lim * SAMPLE_RATE).astype(int)

			if crop_lim[0] > crop_lim[1]:
				print 'ERROR: crop[0] > crop[1]'
				sys.exit()

			if np.sum(crop_lim) > len(self.data_full):
				print 'WARNING: crop out of bounds. len(self.data_full) = {}'\
					.format(len(self.data_full))

			data = self.data_full[crop_lim[0]:crop_lim[1]]

			if self.norm_vol[1]:
				data = self.normalize(data)

			self.data = data


	def _spawn(self, windows_raw):
		windows = []
		kwargs = {
			'fname': self.fname,
			'time_units': self.time_units
		}
		parent_type = type(self)
		for i, w in enumerate(windows_raw):
			kwargs.update({'name': '{} window #{}.'.format(self.name, i)})
			if parent_type is Trajectory:
				windows.append(Trajectory(w, **kwargs))
			elif parent_type is TimeSeries:
				windows.append(TimeSeries(w, **kwargs))
		return windows

	def slice(self, num_windows, window_length):
		if num_windows is None:
			return
		else:
			# start_points = np.floor(
			# 	np.linspace(0, len(self.data) - 1, num_windows, endpoint=False)
			# ).astype(int)

			# toggle start_points to reproduce IDA fig 5
			crop_0, crop_1 = self.crop_lim
			start_points = np.linspace(
				crop_0, crop_1, num_windows, endpoint=False
			)

			if self.time_units == 'samples':
				start_points = start_points.astype(int)

			if window_length is None:
				window_length = len(self.data) / num_windows

			if self.time_units == 'samples':
				start_points_idxs = start_points
			elif self.time_units == 'seconds':
				window_length = int(window_length * SAMPLE_RATE)
				start_points_idxs = (start_points * SAMPLE_RATE).astype(int)

			windows = [self.data[sp:sp + window_length]
			           for sp in start_points_idxs]


		if self.norm_vol[2]:
			windows = [self.normalize(w) for w in windows]

		self.win_start_pts =  start_points
		self.windows = self._spawn(windows)



class TimeSeries(BaseTrajectory):

	def __init__(self, data, **kwargs):
		super(TimeSeries, self).__init__(data, **kwargs)

		self.source_traj = None
		self.project_axis = None

	def embed(self, tau, m):
		if self.time_units == 'seconds':
			tau = int(tau * SAMPLE_RATE)
		data = embed(self.data_full, tau, m)
		traj = Trajectory(
			data,
			fname=self.fname,
			crop=self.crop_lim,
			num_windows=self.num_windows,
			# window_length=self.window_length + tau  # + 1 ??
			window_length=self.window_length,
			vol_norm=self.norm_vol,
			time_units=self.time_units
		)

		traj.source_ts = self
		traj.embed_params = {'tau': tau, 'm': m}

		return traj

	def plot(self, filename):
		plots.ts(self, filename)




class Trajectory(BaseTrajectory):

	def __init__(self, data, **kwargs):
		super(Trajectory, self).__init__(data, **kwargs)

		self.source_ts = None
		self.embed_params = None

		self.dim = self.data.shape[1]

		self.filts = None


	def filtrations(self, filt_params, quiet):
		if self.windows is None:
			print 'ERROR: self.windows is None'
			sys.exit()

		print_title('building filtrations for {}...'.format(self.name))
		filts = []
		for i, t in enumerate(self.windows):
			if quiet:
				print_still(
					'window {} of {}...'.format(i + 1, len(self.windows))
				)
			else:
				print_title(
					'{} window {} of {}...'.format(
						self.name, i + 1, len(self.windows)
					)
				)
			f = Filtration(t, filt_params, silent=quiet,
			               name='{}__window_{}'.format(self.name, i))
			filts.append(f)
		print 'done.'
		self.filts = filts
		return filts


	def project(self, axis=0):
		data = self.data_full[:, axis]

		ts = TimeSeries(
			data,
			fname=self.fname,
			crop=self.crop_lim,
			num_windows=self.num_windows,
			# window_length=self.window_length + tau  # + 1 ??
			window_length=self.window_length,
			vol_norm=self.norm_vol,
			time_units=self.time_units
		)

		ts.source_traj = self
		ts.project_axis = axis

		return ts


