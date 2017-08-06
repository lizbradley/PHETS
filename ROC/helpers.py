import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack, interpolate

from DCE.Plots import plot_signal
from config import WAV_SAMPLE_RATE


def crop_sig(sig, crop):
	if crop is None:
		return sig
	return sig[crop[0] : crop[1]]


def slice_sig(sig, l, n, normalize=True):
	start_pts = np.floor(np.linspace(0, len(sig) - 1, n, endpoint=False)).astype(int)
	windows = np.asarray([sig[pt:pt + l] for pt in start_pts])
	if normalize:
		windows = [np.true_divide(w, np.max(np.abs(w))) for w in windows]

	return windows, start_pts


def get_spec(sig):
	sig_fft = fftpack.rfft(sig)
	spec = 20 * np.log10(np.abs(sig_fft))
	n = sig_fft.size
	timestep = 1 / WAV_SAMPLE_RATE
	freq = fftpack.rfftfreq(n, d=timestep)
	return [freq, spec]



def downsample_spec(freqs, spec, n):
	freqs_interp = np.logspace(1, 4, n)
	spec_interp = interpolate.interp1d(freqs, spec, bounds_error=False, fill_value=0)
	return freqs_interp, spec_interp(freqs_interp)



def plot_sig(sig_full, crop, windows, st_pts, fname):
	print 'plotting signal...'
	fig = plt.figure(figsize=(10, 5), tight_layout=True, dpi=600)

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	plot_signal(ax1, sig_full, crop, 'samples')

	t = np.arange(crop[0], crop[1])
	ax2.plot(t, sig_full[crop[0] : crop[1]], lw=.5, c='k', zorder=0)

	window_st_pts = st_pts + crop[0]

	for pt, w in zip(window_st_pts, windows):
		# ax2.axvline(pt, c='C1', zorder=2)
		# ax2.axvspan(pt, pt + len(w), alpha=.2, zorder=2, color='red')
		t = np.arange(pt, pt + len(w))
		ax2.plot(t, w, zorder=1, c='red', lw=.5, alpha=.5)
		ax2.set_ylim([-1.1, 1.1])
		ax2.yaxis.set_ticks([-1, 0, 1])




	ax1.set_title('full signal')
	ax2.set_title('windows')

	plt.savefig(fname)
	plt.clf()


def plot_spec(spec, fname):
	print 'plotting spectrum...'
	plt.semilogx(*spec, basex=10)
	plt.set_xlim = (10, 50000)
	plt.savefig(fname)
	plt.clf()


def plot_spec_x(spec, fname):
	print 'plotting spectrum x...'
	plt.plot(spec)
	plt.savefig(fname)
	plt.clf()


def plot_roc(ax, data, k, title):
	fpr, tpr = data
	l, = ax.plot(fpr, tpr, clip_on=False, lw=3, zorder=0)
	k = np.arange(*k)
	# k = np.power(np.arange(*k), 2)
	cm = ax.scatter(fpr, tpr, s=100, zorder=1, clip_on=False, c=k, alpha=1)
	ax.plot([0, 1], [0, 1], '--')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	ax.grid()
	ax.set_aspect('equal')
	ax.set_xlabel('false positive rate')
	ax.set_ylabel('true positive rate')
	ax.set_title(title)
	return l, cm
