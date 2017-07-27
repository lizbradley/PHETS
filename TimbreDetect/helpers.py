import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack, interpolate

from config import WAV_SAMPLE_RATE


def crop_sig(sig, crop):
	return sig[crop[0] : crop[1]]


def slice_sig(sig, l=2000, n=25, normalize=True):

	start_pts = np.floor(np.linspace(0, len(sig) - 1, n, endpoint=False)).astype(int)
	windows = np.asarray([sig[pt:pt + l] for pt in start_pts])

	if normalize:
		windows = [np.true_divide(w, np.max(np.abs(w))) for w in windows]

	return windows


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





def plot_sig(sig, sig_full, windows, fname):
	print 'plotting signal...'
	fig = plt.figure(figsize=(10, 5))

	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)

	lw = .1
	ax1.plot(sig_full, lw=lw)
	ax2.plot(sig, lw=lw)
	ax3.plot(np.concatenate(windows), lw=lw)

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


def plot_roc(ax, data, title):
	fpr, tpr = data
	ax.plot(fpr, tpr, clip_on=False, lw=2)
	ax.scatter(fpr, tpr, clip_on=False)
	ax.plot([0, 1], [0, 1], '--')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	ax.grid()
	ax.set_aspect('equal')
	ax.set_xlabel('false positive rate')
	ax.set_ylabel('true positive rate')
	ax.set_title(title)
