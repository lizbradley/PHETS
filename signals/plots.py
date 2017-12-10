import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

from config import SAMPLE_RATE


def _highlight_window(ax, ts, window):
	start_pts = ts.win_start_pts + ts.crop_lim[0]
	start = start_pts[window]
	window_lim = (start, start + ts.window_length)

	if window_lim[1] - window_lim[0] < .005 * (ts.crop_lim[1] - ts.crop_lim[0]):
		ax.axvline(window_lim[0], color='r', alpha=0.7, zorder=1)
	else:
		ax.axvspan(window_lim[0], window_lim[1], fc='r', alpha=0.5, zorder=1)


def ts_full_ax(ax, ts):
	y = ts.data_full

	if ts.time_units == 'samples':
		x = np.arange(0, len(y))
		ax.set_xlabel('time (samples)')

	else: 			# seconds
		x = np.linspace(0, len(y) / SAMPLE_RATE, len(y))
		ax.set_xlabel('time (seconds)')

	ax.plot(x, y, color='k', zorder=0, lw=.5)
	# ax.axis('tight')

	if ts.crop_lim is not None:
		ax.axvspan(x[0], ts.crop_lim[0], color='k', alpha=.3, zorder=1)
		ax.axvspan(ts.crop_lim[1], x[-1], color='k', alpha=.3, zorder=1)

	if ts.num_windows is None:
		n = 0
	else:
		n = ts.num_windows
	for i in range(n):
		_highlight_window(ax, ts, i)

	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))


def ts_ax(ax, ts, show_window='all'):
	y = ts.data
	x = np.linspace(ts.crop_lim[0], ts.crop_lim[1], len(ts.data))

	ax.set_xlabel('time ({})'.format(ts.time_units))

	ax.plot(x, y, color='k', lw=.5)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

	if ts.num_windows is None:
		pass
	elif isinstance(show_window, basestring) and show_window == 'all':
		for i in range(ts.num_windows):
			_highlight_window(ax, ts, i)
	elif isinstance(show_window, int):
		_highlight_window(ax, ts, show_window)



def ts_window_ax(ax, ts, window):
	y = ts.windows[window]
	s = ts.win_start_pts[window]
	x = np.linspace(s, s + ts.window_length)

	ax.plot(x, y, color='k')


def ts_full_fig(ts, out_fname):
	print 'plotting time-series...'
	fig = plt.figure(figsize=(8, 2.5), tight_layout=True, dpi=300)
	ax = fig.add_subplot(111)
	ts_full_ax(ax, ts)
	ax.set_title(ts.name)
	plt.savefig(out_fname)


def ts_fig(ts, out_fname):
	print 'plotting time-series (crop)...'
	fig = plt.figure(figsize=(8, 2.5), tight_layout=True, dpi=300)
	ax = fig.add_subplot(111)
	ts_ax(ax, ts)
	ax.set_title(ts.name)
	plt.savefig(out_fname)


def traj_full_ax(ax, traj):
	data = traj.data_full
	ax.scatter(*data.T, color='black', s=.1)
	ax.set(aspect='equal', adjustable='datalim', anchor='C')


def traj_ax(ax, traj):
	data = traj.data
	ax.scatter(*data.T, color='black', s=.1)
	ax.set(aspect='equal', adjustable='datalim', anchor='C')


def traj_full_fig(traj, out_fname):
	fig = plt.figure(figsize=(6, 5), tight_layout=True, dpi=300)
	if traj.dim == 2:
		ax = fig.add_subplot(111)
	else:
		ax = fig.add_subplot(111, projection='3d')
	traj_full_ax(ax, traj)
	ax.set_title(traj.name)

	plt.savefig(out_fname)
	plt.close(fig)


def traj_fig(traj, out_fname):

	fig = plt.figure(figsize=(6, 5), tight_layout=True, dpi=300)
	if traj.dim == 2:
		ax = fig.add_subplot(111)
	else:
		ax = fig.add_subplot(111, projection='3d')
	traj_ax(ax, traj)
	ax.set_title(traj.name)

	plt.savefig(out_fname)
	plt.close(fig)


