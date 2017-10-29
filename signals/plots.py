import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from config import WAV_SAMPLE_RATE

def ts_zoom(ax, full_sig, crop, time_units='seconds', sig=None):

	if full_sig is None:
		x = np.linspace(crop[0], crop[1], len(sig))
		y = sig

	else:
		# backwards compatibility #
		if time_units == 'samples':
			x = np.arange(crop[0], crop[1])
		elif time_units == 'seconds':
			crop = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
			x = np.linspace(0, len(full_sig) / WAV_SAMPLE_RATE, len(full_sig))[crop[0]:crop[1]]
		y = full_sig[crop[0]:crop[1]]


	ax.plot(x, y, color='k', zorder=0, lw= .5)


def ts(out, waveform_data, window=None, time_units='seconds', offset=None):

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
