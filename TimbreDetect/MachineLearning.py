from __future__ import division

from time import sleep
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import sys
import random

from DCE.Plotter import plot_waveform, plot_waveform_zoom

WAV_SAMPLE_RATE = 44100.



def get_sigs(dir, dir_base, idxs, crop):
	global show_fig
	sigs=[]
	for i in idxs:

		print dir_base, i
		filename = '{}/{:02d}{}'.format(dir, i, dir_base)
		sig = np.loadtxt(filename)

		if show_fig:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			plot_waveform(ax, sig, crop)
			fig.show()
			ans = raw_input('next, continue, or exit? (n/c/q)')
			if ans == 'q': sys.exit()
			elif ans == 'c': show_fig = False

			plt.close()


		crop_samp = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		full_sig = sig
		sig = np.asarray(sig[crop_samp[0]:crop_samp[1]])

		if normalize_volume: sig = sig / np.max(np.abs(sig))

		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# plot_waveform_zoom(ax, full_sig, crop)
		# plt.show()
		# sleep(.5)
		# plt.clf()

		sigs.append(sig)

	show_fig = set_show_fig

	return sigs


def svm_direct(sigs_1, sigs_2, n=10):
	def test(sigs_test, target, clf):
		preds = []
		for sig in sigs_test:
			pred = clf.predict(sig.reshape(1, -1))
			preds.append(pred)

		acc = preds.count(target) / len(sigs_test)
		return acc

	acc_1 = []
	acc_2 = []
	for i in xrange(n):
		np.random.shuffle(sigs_1)
		np.random.shuffle(sigs_2)

		sigs_1_train = sigs_1[::2]
		sigs_1_test = sigs_1[1::2]

		sigs_2_train = sigs_2[::2]
		sigs_2_test = sigs_2[1::2]

		# train #
		train_data = np.concatenate([sigs_1_train, sigs_2_train])
		targets = np.concatenate([[target_1 for sig in sigs_1_train], [target_2 for sig in sigs_2_train]])
		# clf = svm.SVC(verbose=True, probability=True)
		clf = svm.SVC(C=10, gamma=.01, decision_function_shape='ovr')
		clf.fit(train_data, targets)

		# test #
		acc_1.append(test(sigs_1_test, target_1, clf))
		acc_2.append(test(sigs_2_test, target_2, clf))

		del clf

	return acc_1, acc_2



# =================================================================
# 	config
# =================================================================

dir_1 = 'datasets/time_series/C134C'
dir_2 = 'datasets/time_series/viol'

idxs_1 = range(36, 65)
idxs_2 = idxs_1
# idxs_2 = [27, 28, 30, 32, 34, 35, 37, 39, 40, 42, 44, 46, 47, 49, 53]

dir_1_base = '-C134C.txt'
dir_2_base = '-viol.txt'

target_1 = 'C134C'
target_2 = 'viol'

crop_1	 = (.5, 1.5)
crop_2 = crop_1

normalize_volume=True

set_show_fig = False

rebuild = False

# =================================================================
# 	main
# =================================================================

# build / load #
show_fig = set_show_fig
if rebuild:
	sigs_1 = get_sigs(dir_1, dir_1_base, idxs_1, crop_1)
	sigs_2 = get_sigs(dir_2, dir_2_base, idxs_2, crop_2)
	np.save('sigs.npy', [sigs_1, sigs_2])

sigs_1, sigs_2 = np.load('sigs.npy')

# direct #
print 'working...'
acc_1, acc_2 = svm_direct(sigs_1, sigs_2, n=35)

print ''
print 'mean accuracy 1:', np.mean(acc_1)
print acc_1, '\n'
print 'mean accuracy 2:', np.mean(acc_2)
print acc_2, '\n'


