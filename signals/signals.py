import sys
import numpy as np

import plots
from DCE import embed
from PH import Filtration
from config import SAMPLE_RATE
from utilities import print_title, print_still


class CropError(Exception):
	pass

class BaseTrajectory(object):

	def __init__(self,
			data,
			crop=(None, None),
			num_windows=None,
			window_length=None,
			vol_norm=(False, False, False),
	        time_units='samples',
			name=None,
		    fname=None,
	):
		"""
		Parameters
		----------
		data : str or array
			The filename to load, or array. If a filename, sets ``fname``.

		crop : array, optional
			Range of signal to work with. Observes ``time_units``. Either or both
			bounds may be None.
			format: (start, stop).
			default: (None, None)

		num_windows : int, optional
			Slice signal into ``windows`` evenly spaced windows.
			default: None

		window_length : int or float, optional
			Observes ``time_units``
			if None, ``window_length == len(data) / num_windows``
			default: None

		vol_norm : arr, optional
			Normalize amplitude by (full, crop, window).
			default: (False, False, False)

		time_units : str, optional
			``'samples'`` or ``'seconds'``
			Observes ``config.SAMPLE_RATE``
			default: ``'samples'``

		name : string, optional
			Sets ``name``, a label used for titles for plots. If None and
			``fname`` is not None, ``name`` is derived from ``fname``.
			default: None

		fname : string, optional
			If ``data`` is not a filename (i.e. is an array), sets ``fname``.
			default: None
		"""
		if isinstance(data, basestring):        # is filename
			print 'loading input file...'
			self.data_full = np.loadtxt(data)
			self.fname = data
		else:                                   # is array
			self.data_full = data
			self.fname = fname

		if name is not None:
			self.name = name
		elif self.fname is not None:
			self.name = self.fname.split('/')[-1]
		else:
			self.name = None

		self.norm_vol = vol_norm
		self.crop_lim = None
		self.crop_cmd = crop
		self.num_windows = num_windows
		self.window_length = window_length
		self.time_units = time_units

		if self.norm_vol[0]:
			self.data_full = self._normalize(self.data_full)

		self.data = None
		self.windows = None
		self.win_start_pts = None
		self.crop(crop)
		self.slice(num_windows, window_length)


	@staticmethod
	def _normalize(data):
		if np.max(np.abs(data)) == 0:
			pass
		return np.true_divide(data, np.max(np.abs(data)))

	def _to_samples(self, time):
		if self.time_units == 'samples':
			return int(time)
		elif self.time_units == 'seconds':
			return int(time * SAMPLE_RATE)

	def _from_samples(self, time):
		if self.time_units == 'samples':
			return time
		elif self.time_units == 'seconds':
			return time / SAMPLE_RATE

	def crop(self, crop_cmd):
		""" Set ``data`` to the region of ``data_full`` specified by ``crop_cmd`` and
		``time_units``.

		Parameters
		----------
		crop_cmd : array
			observes ``time_units``
			format: (start, stop)

		Returns
		-------

		"""
		to_samples, from_samples = self._to_samples, self._from_samples

		crop_lim = list(crop_cmd)
		if crop_cmd[0] is None:
			crop_lim[0] = 0
		if crop_cmd[1] is None:
				crop_lim[1] = from_samples(len(self.data_full))

		self.crop_lim = crop_lim

		if crop_lim[0] > crop_lim[1]:
			raise CropError('crop[0] > crop[1]')

		if to_samples(crop_lim[1]) > len(self.data_full):
			err = 'crop out of bounds. len(self.data_full) == {}'
			raise CropError(err.format(len(self.data_full)))

		data = self.data_full[to_samples(crop_lim[0]):to_samples(crop_lim[1])]

		if self.norm_vol[1]:
			data = self._normalize(data)

		self.data = data


	def _spawn(self, windows_raw):
		"""
		helper for ``self.slice()``

		Parameters
		----------
		windows_raw : array
			raw window data as an array
			if ``self`` is a ``TimeSeries``
				``windows_raw.shape == (num_windows, window_length)``
			if ``self`` is a ``Trajectory``
				``windows_raw.shape == (num_windows, dim, window_length)``

		Returns
		-------
		windows : array
			an array of windows with the same type as ``self``
		"""
		windows = []
		kwargs = {
			'fname': self.fname,
			'time_units': self.time_units
		}
		parent_type = type(self)
		for i, w in enumerate(windows_raw):
			kwargs.update({'name': '{} window {}'.format(self.name, i)})
			if parent_type is Trajectory:
				windows.append(Trajectory(w, **kwargs))
			elif parent_type is TimeSeries:
				windows.append(TimeSeries(w, **kwargs))
		return windows

	def slice(self, num_windows, window_length=None):
		"""
		Sets ``windows``, an array of evenly spaced windows from ``data``.

		Parameters
		----------
		num_windows: int
		window_length: int or float, optional
			observes 'time_units``
			if None, ``window_length == len(data) / num_windows``
			default: None

		Returns
		-------

		"""
		if num_windows is None:
			return

		crop_0, crop_1 = self.crop_lim
		start_points = np.linspace(
			crop_0, crop_1, num_windows, endpoint=False
		)

		start_points_idxs = [self._to_samples(s) for s in start_points]

		if window_length is None:
			window_length_samp = int(len(self.data) / num_windows)
			window_length = self._from_samples(window_length_samp)
		else:
			window_length_samp = self._to_samples(window_length)

		windows = [self.data_full[sp:sp + window_length_samp]
		           for sp in start_points_idxs]

		if self.norm_vol[2]:
			windows = [self._normalize(w) for w in windows]

		self.window_length = window_length
		self.win_start_pts = start_points
		self.windows = self._spawn(windows)



class TimeSeries(BaseTrajectory):

	def __init__(self, data, **kwargs):
		""" See :py:class:`BaseTrajectory` for parameter descriptions """
		super(TimeSeries, self).__init__(data, **kwargs)

		self.source_traj = None
		self.project_axis = None

	def embed(self, tau, m):
		"""
		Embed ``data_full``, re-apply crop and slicing.

		Parameters
		----------
		tau : int or float
			observes ``time_units``
		m : int


		Returns
		-------
		Trajectory

		"""
		if self.time_units == 'seconds':
			tau = int(tau * SAMPLE_RATE)
		data = embed(self.data_full, tau, m)
		traj = Trajectory(
			data,
			fname=self.fname,
			crop=self.crop_cmd,
			num_windows=self.num_windows,
			window_length=self.window_length,
			vol_norm=self.norm_vol,
			time_units=self.time_units
		)

		traj.source_ts = self
		traj.embed_params = {'tau': tau, 'm': m}

		return traj

	def plot(self, filename):
		"""
		Plot full time series with crop and windows demarcated, save to
		``filename``.

		Parameters
		----------
		filename : str

		Returns
		-------

		"""
		plots.ts_fig(self, filename)




class Trajectory(BaseTrajectory):

	def __init__(self, data, **kwargs):
		""" See :py:class:`BaseTrajectory` for parameter descriptions """
		super(Trajectory, self).__init__(data, **kwargs)

		self.source_ts = None
		self.embed_params = None

		self.dim = self.data.shape[1]

		self.filts = None


	def filtrations(self, filt_params, quiet):
		""" Compute filtration for each window of trajectory.

		Parameters
		----------
		filt_params : dict
			see Filtration
		quiet : bool
			terminal output noise

		Returns
		-------
		array
			array of Filtration objects

		"""
		if self.windows is None:
			raise Exception('self.windows is None')

		print_title('building filtrations for {}...'.format(self.name))
		filts = []
		for i, traj in enumerate(self.windows):
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
			f = Filtration(traj, filt_params, silent=quiet, save=False)
			filts.append(f)
		print 'done.'
		self.filts = filts
		return filts


	def project(self, axis=0):
		""" Project ``self.data_full`` to time series, re-apply crop and slicing.

		Parameters
		----------
		axis : int

		Returns
		-------
		TimeSeries

		"""
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


