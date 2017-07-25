import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm

import matplotlib.pyplot as plt
from config import WAV_SAMPLE_RATE

def crop_sig(sig, crop):
	return sig[crop[0] : crop[1]]


def slice_sig(sig, l=2000, n=25, normalize=True):

	start_pts = np.floor(np.linspace(0, len(sig) - 1, n, endpoint=False)).astype(int)
	windows = np.asarray([sig[pt:pt + l] for pt in start_pts])

	if normalize:
		windows = [np.true_divide(w, np.max(np.abs(w))) for w in windows]

	return windows


from scipy import fftpack

def get_spec(sig):
	sig_fft = fftpack.rfft(sig)

	spec = 20 * np.log10(np.abs(sig_fft))
	n = sig_fft.size
	timestep = 1 / WAV_SAMPLE_RATE
	freq = fftpack.rfftfreq(n, d=timestep)

	return [freq, spec]

from scipy import interpolate

def downsample_spec(freqs, spec, n):
	freqs_interp = np.logspace(1, 4, n)
	spec_interp = interpolate.interp1d(freqs, spec, bounds_error=False, fill_value=0)
	return freqs_interp, spec_interp(freqs_interp)




def plot_sig(sig, sig_full, windows, fname):
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


fname_1 = '../datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt'
fname_2 = '../datasets/time_series/viol/40-viol.txt'

label_1 = 'clarinet'
label_2 = 'viol'

crop_1 = (50000, 120000)
crop_2 = (100000, 170000)

window_length = 5000
num_windows = 50

FT_bins = 50




print 'loading...'
sig_full_1 = np.loadtxt(fname_1)
sig_full_2 = np.loadtxt(fname_2)

print 'processing...'
sig_1 = crop_sig(sig_full_1, crop_1)
sig_2 = crop_sig(sig_full_2, crop_2)


windows_1 = slice_sig(sig_1, l=window_length, n=num_windows)
windows_2 = slice_sig(sig_2, l=window_length, n=num_windows)

print 'plotting...'

plot_sig(sig_1, sig_full_1, windows_1, label_1)
plot_sig(sig_2, sig_full_2, windows_2, label_2)


specs_1 = [get_spec(w) for w in windows_1]
specs_2 = [get_spec(w) for w in windows_2]


plt.semilogx(*specs_1[0], basex=10)
plt.set_xlim=(10, 50000)
plt.savefig('spec.png')
plt.clf()

specs_ds_1 = np.asarray([downsample_spec(s[0], s[1], FT_bins) for s in specs_1])
specs_ds_2 = np.asarray([downsample_spec(s[0], s[1], FT_bins) for s in specs_2])

plt.semilogx(*specs_ds_1[0], basex=10)
plt.set_xlim=(10, 50000)
plt.savefig('spec_ds.png')
plt.clf()

specs = np.concatenate([specs_ds_1[:, 1], specs_ds_2[:, 1]])
labels = [label_1 for s in specs_ds_1] + [label_2 for s in specs_ds_2]


train_specs, test_specs, train_labels, test_labels = train_test_split(specs, labels, train_size=0.8)

clf = svm.SVC()
print 'training...'
clf.fit(train_specs, train_labels)

print 'testing...'
print clf.score(test_specs, test_labels)