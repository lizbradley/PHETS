from time import sleep
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import sys
import random

import scipy.fftpack
from scipy import interpolate


from DCE.Plots import plot_signal, plot_signal_zoom

WAV_SAMPLE_RATE = 44100.

class Signal:
	def __init__(self, arr, idx, crop, inst_type):
		self.arr_full = np.array(arr)
		self.set_crop(crop)
		self.idx = idx
		self.len = len(self.arr_crop)
		self.inst_type = inst_type

	def set_crop(self, crop):
		self.crop = np.array(crop)
		crop_samp = (self.crop * WAV_SAMPLE_RATE).astype(int)
		self.arr_crop = self.arr_full[crop_samp[0]: crop_samp[1]]

		if normalize_volume:
			self.arr_crop = self.arr_crop / np.max(self.arr_crop)



def get_sigs(dir, dir_base, idxs, crop, inst_type):
	sigs=[]
	for i in idxs:
		print dir_base, i
		filename = '{}/{:02d}{}'.format(dir, i, dir_base)
		sig_arr = np.loadtxt(filename)

		sig = Signal(sig_arr, i, crop, inst_type)
		sigs.append(sig)

	return sigs

def recrop_sigs(sigs, crop):
	for sig in sigs:
		sig.set_crop(crop)


def get_spec(sig):

	FFT_x = scipy.fftpack.fftfreq(sig.len, d=1 / WAV_SAMPLE_RATE)

	FFT = scipy.fftpack.fft(sig.arr_crop)
	FFT = 20 * scipy.log10(scipy.absolute(FFT)) 	# convert to db

	FFT_pos = FFT[1:len(FFT)/2]

	spec_x = FFT_x[1:len(FFT_x)/2]

	spec = FFT_pos

	return spec_x, spec


def normalize_spec(idx, freqs, spec):
	# freqs_interp = np.linspace(.01, 100, FT_res)
	freqs_interp = np.logspace(1, 4, n)

	ff = np.power(2, (40 - float(idx)) / 12) * 440  # Hz, descending index
	freqs = freqs / ff

	spec_interp = interpolate.interp1d(freqs, spec, bounds_error=False, fill_value=np.nan)

	return freqs_interp, spec_interp(freqs_interp)


def get_mean_spec(sigs):
	specs = []
	freqs = []
	for sig in sigs:

		freqs, spec = get_spec(sig)

		freqs, spec = normalize_spec(sig.idx, freqs, spec)

		specs.append(spec)

	return freqs, np.mean(specs, axis=0)


def plot_spec(freqs, spec, out_filename):
	fig = plt.figure(figsize=(10, 4))
	ax = fig.add_subplot(111)
	ax.semilogx(freqs, spec, lw=1, c='k')
	# ax.set_xticks(np.arange(0, 50, 2))
	ax.grid()
	plt.savefig(out_filename)


def test_FT_detect(sigs, mean_spec_1, mean_spec_2):
	corr = []
	for sig in sigs:
		freqs, spec = get_spec(sig)
		freqs, spec = normalize_spec(sig.idx, freqs, spec)

		diffs_vs_1 = np.subtract(spec, mean_spec_1)
		dist_vs_1 = np.power(np.nansum(np.power(diffs_vs_1, 2)), .5)

		diffs_vs_2 = np.subtract(spec, mean_spec_2)
		dist_vs_2 = np.power(np.nansum(np.power(diffs_vs_2, 2)), .5)

		print 'idx:', sig.idx
		print '\t dist vs 1:', dist_vs_1
		print '\t dist vs 2:', dist_vs_2
		if dist_vs_2 > dist_vs_1:
			print 'result: 1 -- ' + target_1
			result = target_1
		elif dist_vs_1 > dist_vs_2:
			print 'result: 2 -- ' + target_2
			result = target_2
		else:
			print 'result: unknown'
			result = None
		print ''

		corr.append(result)
	return corr


# =================================================================
# 	config
# =================================================================

dir_1 = 'datasets/time_series/viol'
dir_2 = 'datasets/time_series/C134C'

idxs_1 = np.arange(36, 65)
idxs_2 = idxs_1
# idxs_2 = [27, 28, 30, 32, 34, 35, 37, 39, 40, 42, 44, 46, 47, 49, 53]

dir_1_base = '-viol.txt'
dir_2_base = '-C134C.txt'

target_1 = 'viol'
target_2 = 'C134C'

crop_1 = (.5, 1)
crop_2 = crop_1

FT_res = 1000

normalize_volume = True

set_show_fig = False

reload_sigs = False

# =================================================================
# 	main
# =================================================================


# build / load #
show_fig = set_show_fig
if reload_sigs:
	sigs_1 = get_sigs(dir_1, dir_1_base, idxs_1, crop_1, target_1)
	sigs_2 = get_sigs(dir_2, dir_2_base, idxs_2, crop_2, target_2)
	np.save('TimbreDetect/sigs.npy', [sigs_1, sigs_2])


sigs_1, sigs_2 = np.load('TimbreDetect/sigs.npy')

# sigs_1, sigs_2 = sigs_2, sigs_1
#
# recrop_sigs(sigs_1, crop_1)
# recrop_sigs(sigs_2, crop_2)


acc_1 = []
acc_2 = []
acc_0 = []
for i in xrange(30):
	np.random.shuffle(sigs_1)
	np.random.shuffle(sigs_2)

	sigs_1_train = sigs_1[::2]
	sigs_2_train = sigs_2[::2]

	sigs_1_test = sigs_1[1::2]
	sigs_2_test = sigs_2[1::2]

	freqs, mean_spec_1 = get_mean_spec(sigs_1_train)
	freqs, mean_spec_2 = get_mean_spec(sigs_2_train)


	results_1 = test_FT_detect(sigs_1_test, mean_spec_1, mean_spec_2)
	results_2 = test_FT_detect(sigs_2_test, mean_spec_1, mean_spec_2)

	corr_1 = results_1.count(target_1)
	corr_2 = results_2.count(target_2)
	corr_0 = results_1.count(None) + results_2.count(None)

	print 'results 1:', corr_1, ' / ', len(results_1)
	print 'results 2:', corr_2, ' / ', len(results_2)
	print 'unknown:', corr_0

	acc_1.append(corr_1)
	acc_2.append(corr_2)
	acc_0.append(corr_0)

print ''
print 'avg test 1 correct:', np.mean(acc_1)
print acc_1
print ''
print 'avg test 2 correct:', np.mean(acc_2)
print acc_2
print ''
print 'avg unknown:', np.mean(acc_0)

