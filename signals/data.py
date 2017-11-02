import numpy as np
import sys

from DCE import embed
from PH import Filtration
from config import SAMPLE_RATE
from utilities import print_title


class BaseTrajectory(object):

	def __init__(self,
	        data,
	        fname=None,
			crop=None,
			num_windows=None,
			window_length=None,
			vol_norm=(False, False, False),     # (full, crop, windows)
	        time_units='samples'


	):
		if isinstance(data, basestring):        # is filename
			self.data_full = np.loadtxt(data)
			self.fname = data
		else:                                   # is array
			self.data_full = data
			self.fname = fname

		if self.fname is not None:
			self.name = self.fname.split('/')[-1].split('.')[0]
		else:
			self.name = None

		self.norm_vol = vol_norm
		self.crop_lim = crop
		self.num_windows = num_windows
		self.window_length = window_length
		if time_units in ('samples', 'seconds'):
			self.time_units = time_units
		else:
			print "ERROR: invalid 'time_units'; use 'samples' or 'seconds'"
			sys.exit()

		if self.norm_vol[0]:
			self.data_full = self.normalize(self.data_full)
		self.data = self.crop(crop)
		self.windows, self.win_start_idxs = self.slice(
			num_windows, window_length
		)


	@staticmethod
	def normalize(data):
		return np.true_divide(data, np.max(np.abs(data)))


	def crop(self, lim):
		if lim is None:
			data = self.data_full

		else:
			crop_lim = np.array(self.crop_lim)
			self.crop_lim = crop_lim

			if self.time_units == 'seconds':
				crop_lim = (crop_lim * SAMPLE_RATE).astype(int)

			if np.sum(crop_lim) > len(self.data_full):
				print 'WARNING: crop out of bounds'

			data = self.data_full[crop_lim[0]:crop_lim[1]]

			if self.norm_vol[1]:
				data = self.normalize(data)

		self.data = data
		return data


	def slice(self, num_windows, window_length):
		if num_windows is None:
			return None, None
		else:
			start_idxs = np.floor(
				np.linspace(0, len(self.data), num_windows, endpoint=False)
			).astype(int)

			if window_length is None:
				window_length = len(self.data) / num_windows

			if self.time_units == 'seconds':
				window_length = int(window_length * SAMPLE_RATE)
			windows = [self.data[sp:sp + window_length] for sp in start_idxs]

		if self.norm_vol[2]:
			windows = [self.normalize(w) for w in windows]

		self.windows, self.win_start_idxs = windows, start_idxs
		return windows, start_idxs



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




class Trajectory(BaseTrajectory):

	def __init__(self, data, **kwargs):
		super(Trajectory, self).__init__(data, **kwargs)

		self.source_ts = None
		self.embed_params = None

		self.filts = None


	def filtrations(self, filt_params, quiet):
		if self.windows is None:
			print 'ERROR: self.windows is None'
			sys.exit()

		print_title('building filtrations for {}...'.format(self.name))
		filts = []
		for i, t in enumerate(self.windows):
			if quiet:
				sys.stdout.write('\rwindow {} of {}...'.format(
					i + 1, len(self.windows))
				)
				sys.stdout.flush()
			else:
				print_title('{} window {} of {}...'.format(
						self.name, i + 1, len(self.windows))
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


