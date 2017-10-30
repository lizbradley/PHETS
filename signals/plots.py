import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from config import WAV_SAMPLE_RATE

def ts_zoom(ax, ts):

	x = np.linspace(ts.crop_lim[0], ts.crop_lim[1], len(ts.data))
	y = ts.data
	ax.plot(x, y, color='k', zorder=0, lw= .5)


def ts(out, waveform_data, window=None, time_units='seconds', offset=None):
	"""
	plots full signal with crop section highlighted in red
	refactor me!
	"""

	if isinstance(out, basestring):
		fig = plt.figure(figsize=(4, 1))
		ax = fig.add_subplot(111)
	else:
		ax = out

	y = waveform_data

	if time_units == 'samples':
		x = np.arange(0, len(y))
		ax.set_xlabel('time (samples)')

	else: 			# seconds
		x = np.linspace(0, len(y) / WAV_SAMPLE_RATE, len(y))
		ax.set_xlabel('time (seconds)')

	if offset:
		x = x + offset
		window = np.array(window) + offset

	ax.plot(x, y, color='k', zorder=0, lw= .5)
	ax.axis('tight')

	if window is not None:
		if np.abs(window[0] - window[1]) < .01:
			ax.axvline(window[0], color='r', alpha=0.7, zorder=1)
		ax.axvspan(window[0], window[1], facecolor='r', alpha=0.5, zorder=1)

	ymin, ymax = ax.get_ylim()
	ylim = abs(ymin) if abs(ymin) >= abs(ymax) else abs(ymax)
	ax.set_ylim([-ylim, ylim])
	ax.set_yticks([-ylim, 0, ylim])
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))


	if isinstance(out, basestring): plt.savefig(out)
