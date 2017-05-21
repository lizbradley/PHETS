from time import sleep
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import sys
import random

import scipy.fftpack
from scipy import interpolate


from DCE.Plotter import plot_waveform, plot_waveform_zoom

WAV_SAMPLE_RATE = 44100.


def get_sigs(dir, dir_base, idxs, crop):
	sigs=[]
	for i in idxs:

		print dir_base, i
		filename = '{}/{:02d}{}'.format(dir, i, dir_base)
		sig = np.loadtxt(filename)

		sigs.append(sig)

	show_fig = set_show_fig

	return sigs

def crop_sigs(sigs, crop):
	global show_fig

	cropped = []

	for sig in sigs:

		if show_fig:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			plot_waveform(ax, sig, crop)
			fig.show()
			ans = raw_input('next, continue, or exit? (n/c/q)')
			if ans == 'q':
				sys.exit()
			elif ans == 'c':
				show_fig = False

			plt.close()

		crop_samp = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		sig = np.asarray(sig[crop_samp[0]:crop_samp[1]])

		if normalize_volume: sig = sig / np.max(np.abs(sig))

		cropped.append(sig)

	show_fig = set_show_fig

	return np.asarray(cropped)

def get_spec(sig):

	FFT_x = scipy.fftpack.fftfreq(len(sig), d=1 / WAV_SAMPLE_RATE)

	FFT = scipy.fftpack.fft(sig)
	FFT = 20 * scipy.log10(scipy.absolute(FFT)) 	# convert to db

	FFT_pos = FFT[1:len(FFT)/2]

	spec_x = FFT_x[1:len(FFT_x)/2]

	spec = FFT_pos

	return spec_x, spec

def normalize_spec(idx, freqs, spec):
	freqs_interp = np.linspace(.01, 100, 1000)
	ff = np.power(2, (40 - float(idx)) / 12) * 440  # Hz, descending index
	freqs = freqs / ff

	spec_interp = interpolate.interp1d(freqs, spec, bounds_error=False, fill_value=np.nan)

	return freqs_interp, spec_interp(freqs_interp)

def get_mean_spec(idxs, sigs):
	specs = []
	freqs = []
	for idx, sig in zip(idxs, sigs):

		freqs, spec = get_spec(sig)

		freqs, spec = normalize_spec(idx, freqs, spec)

		specs.append(spec)

	return freqs, np.mean(specs, axis=0)

def plot_spec(freqs, spec, out_filename):
	fig = plt.figure(figsize=(10, 4))
	ax = fig.add_subplot(111)
	ax.semilogx(freqs, spec, lw=1, c='k')
	# ax.set_xticks(np.arange(0, 50, 2))
	ax.grid()
	plt.savefig(out_filename)

def test(idxs, sigs):
	corr = []
	for idx, sig in zip(idxs, sigs):
		freqs, spec = get_spec(sig)
		freqs, spec = normalize_spec(idx, freqs, spec)

		diffs_vs_1 = np.subtract(spec, mean_spec_1)
		dist_vs_1 = np.power(np.nansum(np.power(diffs_vs_1, 2)), .5)

		diffs_vs_2 = np.subtract(spec, mean_spec_2)
		dist_vs_2 = np.power(np.nansum(np.power(diffs_vs_2, 2)), .5)

		print 'idx:', idx
		print '\t dist vs 1:', dist_vs_1
		print '\t dist vs 2:', dist_vs_2
		if dist_vs_2 > dist_vs_1:
			print 'result 1'
			result = 1
		elif dist_vs_1 > dist_vs_2:
			print 'result: 2'
			result = 2
		else:
			print 'result: unknown'
			result = 0
		print ''

		corr.append(result)
	return corr


# =================================================================
# 	config
# =================================================================

dir_1 = 'datasets/time_series/C134C'
dir_2 = 'datasets/time_series/C135B'

idxs_1 = np.arange(36, 65)
idxs_2 = idxs_1
# idxs_2 = [27, 28, 30, 32, 34, 35, 37, 39, 40, 42, 44, 46, 47, 49, 53]

dir_1_base = '-C134C.txt'
dir_2_base = '-C135B.txt'

target_1 = 'C134C'
target_2 = 'C135B'

crop_1 = (.5, 1)
crop_2 = crop_1

normalize_volume = True

set_show_fig = False

reload_sigs = True

# =================================================================
# 	main
# =================================================================

# build / load #
show_fig = set_show_fig
if reload_sigs:
	sigs_1 = get_sigs(dir_1, dir_1_base, idxs_1, crop_1)
	sigs_2 = get_sigs(dir_2, dir_2_base, idxs_2, crop_2)
	np.save('TimbreDetect/sigs.npy', [sigs_1, sigs_2])

sigs_1, sigs_2 = np.load('TimbreDetect/sigs.npy')

sigs_1 = crop_sigs(sigs_1, crop_1)
sigs_2 = crop_sigs(sigs_2, crop_2)

sigs_1_train = sigs_1[::2]
sigs_1_test = sigs_1[1::2]

sigs_2_train = sigs_2[::2]
sigs_2_test = sigs_2[1::2]

train_idxs = idxs_1[::2]

freqs, mean_spec_1 = get_mean_spec(train_idxs, sigs_1_train)
freqs, mean_spec_2 = get_mean_spec(train_idxs, sigs_2_train)


results_1 = test(train_idxs, sigs_1_test)
results_2 = test(train_idxs, sigs_2_test)

print 'results 1:', results_1.count(1), ' / ', len(results_1)
print 'results 2:', results_2.count(2), ' / ', len(results_2)
print 'unknown:', results_1.count(0) + results_2.count(0)