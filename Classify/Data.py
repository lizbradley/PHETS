
import time

import numpy as np

from DCE.DCE import embed
from PH import Filtration
from PubPlots import plot_filtration_pub
from Utilities import print_title
from config import default_filtration_params as filt_params
from helpers import crop_sig, slice_sig, get_spec, downsample_spec, plot_sig, plot_roc



def norm(f):
	if len(f.shape) == 1:
		res = len(f)
	elif len(f.shape) == 2:
		# res = len(f) ** .5
		res = len(f)
	else:
		res = None
		print 'ERROR'


	dA = 2. / (res ** 2)		# normalize such that area of PRF domain is 1
	# dA = 1. / ((res - 1) ** 2)
	# dA = 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)


def test_inclusion(mean, var, tests, k):
	var_norm = norm(var)

	pred = []
	for spec in tests:
		diff = np.subtract(spec, mean)

		dist = norm(diff)
		pred.append(dist <= var_norm * k)
	return pred



def get_specs(windows, FT_bins, mode):
	specs = [get_spec(w) for w in windows]
	# plot_spec(specs_1[0], 'spec.png')
	specs = np.asarray([downsample_spec(s[0], s[1], FT_bins, mode)[1] for s in specs])
	# plot_spec(specs_ds_1[0], 'spec_ds.png')
	return specs

from PH import make_PRF_plot
from Utilities import clear_dir


def get_prfs(trajs, filt_params, window_length, num_landmarks, label='none', load_saved=False):

	fname = 'ROC/data/{}_{}.npy'.format(label, window_length)

	if load_saved:
		return np.load(fname)

	res = filt_params['num_divisions']
	filts = []


	for i, t in enumerate(trajs):

		print_title('window_length: {} \t label: {} \t window #: {}'.format(
			window_length, label, i))

		filt_params.update({'worm_length': window_length})
		filt_params.update({'ds_rate': window_length / num_landmarks})
		filt = Filtration(t, filt_params)

		fname_filt = 'output/ROC/samps/{}__filt__wl_{}__win_{}.png'.format(label, window_length, i)
		fname_prf = 'output/ROC/samps/{}__prf__wl_{}__win_{}.png'.format(label, window_length, i)

		# make_movie(filt, fname_movie)
		filt_frame = filt_params['num_divisions'] - 1
		# filt_frame = 2
		plot_filtration_pub(filt, filt_frame, fname_filt)
		make_PRF_plot(filt, fname_prf, annot_hm=False)

		filts.append(filt)

	prfs = [f.get_PRF() for f in filts]
	prfs = [prf[2] for prf in prfs]
	np.save(fname, prfs)
	return prfs



def pre_proc_data(samps):
	samps_train = samps[1::2]
	samps_test = samps[::2]
	mean_samp = np.mean(samps_train, axis=0)
	var_samp = np.var(samps_train, axis=0)
	return samps_train, samps_test, mean_samp, var_samp



def roc_data(mean, var, tests_true, tests_false, k_arr):
	tpr = []
	fpr = []
	for k in k_arr:
		true_pos = test_inclusion(mean, var, tests_true, k)
		false_pos = test_inclusion(mean, var, tests_false, k)
		true_pos_rate = sum(true_pos) / float(len(true_pos))
		false_pos_rate = sum(false_pos) / float(len(false_pos))
		tpr.append(true_pos_rate)
		fpr.append(false_pos_rate)

	return [fpr, tpr]



def roc_title(label, mode):
	return 'is it a {}? [{}]'.format(label, mode)

def letter_label(ax, label, nudge_r=0.):
	ax.text(
		.95 + nudge_r, .1, label,
		size='large',
		horizontalalignment='right',
		verticalalignment='top',
		transform=ax.transAxes,
		bbox=dict(
			alpha=1,
			facecolor='white',
			edgecolor='black',
			pad=5,
			boxstyle='round, pad=.5'
		)
	)



def print_stats(mean_1, var_1, label_1, mean_2, var_2, label_2):
	print 'mean {:10}\t{:06.3f}\t\t\tmean {:10}\t{:06.3f}'.format(
		label_1, norm(mean_1), label_2, norm(mean_2))
	print 'var  {:10}\t{:06.3f}\t\t\tvar {:10}\t{:06.3f}'.format(
		label_1, norm(var_1), label_2, norm(var_2))


def print_stats_multi(spec_data, prf_data, window_length):
	for wl, d1, d2 in zip(window_length, spec_data, prf_data):
		print_title('')
		print_title(wl)
		print_title('FFT:')
		print_stats(*d1)
		print_title('PRF:')
		print_stats(*d2)



class MPRFL2:

	def __init__(self,
		train_data,		# windowed TimeSeries or ndarray
		label,

		tau=None,
		m=None,

	):
		""" classifier which calculates and compares l2 distance to a mean prf """





		if len(windows_1[0].shape) == 1:
			trajs_1 = [embed(w, tau, m) for w in windows_1]
		else:
			trajs_1 = windows_1


		prfs_1 = get_prfs(trajs_1, filt_params, window_length, num_landmarks,
						  load_saved=load_saved_filts, label=label)


		prfs_train_1, prfs_test_1, mean_prf_1, var_prf_1 = pre_proc_data(prfs_1)



		prf_train_dists_1_vs_1 = get_dists(mean_prf_1, prfs_train_1)




		print_stats_multi(print_data_spec, print_data_prf, window_length)

	@staticmethod
	def get_dists(mean, tests):
		return [norm(np.subtract(test, mean)) for test in tests]

if __name__  == '__main__':
	sig = TimeSeries('datasets/time_series/viol/40-viol.txt')

	clf = MRFL2(train=sig.windows())