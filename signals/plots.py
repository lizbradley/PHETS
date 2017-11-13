import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter

from config import SAMPLE_RATE


def highlight_window(ax, ts, window):
	start = ts.win_start_pts[window]
	window_lim = (start, start + ts.window_length)

	if np.abs(window_lim[0] - window_lim[1]) < .01 * len(ts.data):
		ax.axvline(window_lim[0], color='r', alpha=0.7, zorder=1)
	else:
		ax.axvspan(window_lim[0], window_lim[1], fc='r', alpha=0.5, zorder=1)


def ts_ax(ax, ts):
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

	for i in range(ts.num_windows):
		highlight_window(ax, ts, i)

	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

def ts(ts, out_fname):
	fig = plt.figure(figsize=(8, 2.5), tight_layout=True)
	ax = fig.add_subplot(111)
	ts_ax(ax, ts)
	plt.savefig(out_fname)


def ts_crop_ax(ax, ts, show_window='all'):
	y = ts.data
	x = np.linspace(ts.crop_lim[0], ts.crop_lim[1], len(ts.data))

	if ts.time_units == 'samples':
		ax.set_xlabel('time (samples)')
	elif ts.time_units == 'seconds':
		ax.set_xlabel('time (seconds)')

	ax.plot(x, y, color='k')
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

	if isinstance(show_window, basestring) and show_window == 'all':
		for i in range(ts.num_windows):
			highlight_window(ax, ts, i)
	elif isinstance(show_window, int):
		try:
			highlight_window(ax, ts, show_window)
		except IndexError:
			print 'ERROR: signals.plots.ts_crop_ax: show_window out of range'
			sys.exit()



def ts_window_ax(ax, ts, window):
	y = ts.windows[window]
	s = ts.win_start_pts[window]
	x = np.linspace(s, s + ts.window_length)

	ax.plot(x, y, color='k')
