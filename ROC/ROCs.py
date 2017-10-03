import time

import matplotlib
import matplotlib.pyplot as plt
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


def plot_fig(data, k, label_1, label_2, window_length, fname, pub):
	fig = plt.figure(figsize=(10, 8), tight_layout=True, dpi=100)

	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)

	letter_label(ax1, ' a ')
	letter_label(ax2, ' b ')
	letter_label(ax3, ' c ')
	letter_label(ax4, ' d ')


	fig.subplots_adjust(right=0.8)
	cax = fig.add_axes([0.92, 0.05, 0.03, 0.9])

	lines = []


	for data_wl in data:
		fft_data_1, fft_data_2, prf_data_1, prf_data_2 = data_wl

		plot_roc(ax3, fft_data_1, k, roc_title(label_1, 'FFT'))
		plot_roc(ax4, fft_data_2, k, roc_title(label_2, 'FFT'))
		plot_roc(ax1, prf_data_1, k, roc_title(label_1, 'PRF'))
		l, cm = plot_roc(ax2, prf_data_2, k, roc_title(label_2, 'PRF'))
		lines.append(l)

	bounds = np.arange(k[0], k[1] + .5, .5)
	# bounds = np.arange(k[0], k[1] + 1, 1)
	cb = fig.colorbar(cm, cax=cax, boundaries=bounds)
	cb.set_label("$k$", labelpad=-1, size=19)

	labels = bounds[::2]
	loc = labels + .25
	# loc = labels + .5
	cb.set_ticks(loc)
	cb.set_ticklabels([int(l) for l in labels])
	cb.ax.tick_params(labelsize=14)

	if not pub:
		fig.suptitle('k = range({}, {}, {})'.format(*k), fontsize=16)
		fig.legend(lines, labels=window_length)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	plt.savefig(fname)


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







##### set threshold type on line 337 #####


def PRF_vs_FFT_v2(
		fname_1,
		fname_2,
		out_filename,
		label_1,
		label_2,
		crop_1=None,
		crop_2=None,

		tau=None,
		m=2,

		window_length=2000,
		num_windows=15,
		num_landmarks=30,

		FT_bins=50,
		FT_bin_mode='log',

		k=(0, 1, .01),
		pub=False,		# for publication

		load_saved_filts=False,
		normalize_volume=True,

		save_samps=True

):

	def get_dists(mean, tests):
		return [norm(np.subtract(test, mean)) for test in tests]

	def get_rate_vs_k(dists, dists_train, k_arr, mean_train):
		variance = np.mean(np.power(dists_train, 2))
		std_dev = np.mean(np.power(dists_train, 2)) ** .5
		scaled_variance = variance / norm(mean_train)
		rate = []
		for k in k_arr:
			# pred = dists <= variance * k
			# pred = dists <= std_dev * k
			pred = dists <= std_dev * k
			rate.append(sum(pred) / float(len(pred)))
		return rate

	def plot_dists(test_dists, train_dists, var, fname):

		fig = plt.figure(figsize=(10, 6))
		ax = fig.add_subplot(111)
		# ax.set_ylim(bottom=0)

		ax.plot(test_dists, 'o')

		variance = np.mean(np.power(train_dists, 2))

		npv_line = ax.axhline(norm(var), c='C0')
		std_dev_line = ax.axhline(variance ** .5, c='C1')
		var_line = ax.axhline(variance, c='C2')

		fig.legend([npv_line, std_dev_line, var_line],
				   ['norm of pointwise variance', 'standard deviation', 'variance'])



		plt.savefig(fname)


	if save_samps and not load_saved_filts: clear_dir('output/ROC/samps/')
	start = time.time()
	print 'loading signal...'
	sig_full_1 = np.loadtxt(fname_1)
	sig_full_2 = np.loadtxt(fname_2)

	print 'processing...'
	sig_1 = crop_sig(sig_full_1, crop_1)
	sig_2 = crop_sig(sig_full_2, crop_2)

	data = []
	print_data_spec = []
	print_data_prf = []

	if isinstance(window_length, int):
		window_length = [window_length]

	k_arr = np.arange(*k)
	for wl in window_length:
		windows_1, st_pts_1 = slice_sig(sig_1, l=wl, n=num_windows, normalize=normalize_volume)
		windows_2, st_pts_2 = slice_sig(sig_2, l=wl, n=num_windows, normalize=normalize_volume)

		if save_samps and not pub:
			plot_sig(sig_full_1, crop_1, windows_1, st_pts_1, 'output/ROC/sig1_wl{}.png'.format(wl))
			plot_sig(sig_full_2, crop_2, windows_2, st_pts_2, 'output/ROC/sig2_wl{}.png'.format(wl))

		specs_1 = get_specs(windows_1, FT_bins, FT_bin_mode)
		specs_2 = get_specs(windows_2, FT_bins, FT_bin_mode)

		trajs_1 = [embed(w, tau, m) for w in windows_1]
		trajs_2 = [embed(w, tau, m) for w in windows_2]

		prfs_1 = get_prfs(trajs_1, filt_params, wl, num_landmarks, load_saved=load_saved_filts, label=label_1)
		prfs_2 = get_prfs(trajs_2, filt_params, wl, num_landmarks, load_saved=load_saved_filts, label=label_2)

		specs_train_1, specs_test_1, mean_spec_1, var_spec_1 = pre_proc_data(specs_1)
		specs_train_2, specs_test_2, mean_spec_2, var_spec_2 = pre_proc_data(specs_2)

		prfs_train_1, prfs_test_1, mean_prf_1, var_prf_1 = pre_proc_data(prfs_1)
		prfs_train_2, prfs_test_2, mean_prf_2, var_prf_2 = pre_proc_data(prfs_2)



		# PRF STUFF #

		prf_dists_1_vs_1 = get_dists(mean_prf_1, prfs_test_1)
		prf_dists_2_vs_1 = get_dists(mean_prf_1, prfs_test_2)
		prf_dists_1_vs_2 = get_dists(mean_prf_2, prfs_test_1)
		prf_dists_2_vs_2 = get_dists(mean_prf_2, prfs_test_2)


		prf_train_dists_1_vs_1 = get_dists(mean_prf_1, prfs_train_1)
		prf_train_dists_2_vs_2 = get_dists(mean_prf_2, prfs_train_2)


		prf_1_tpr = get_rate_vs_k(prf_dists_1_vs_1, prf_train_dists_1_vs_1, k_arr, mean_prf_1)
		prf_1_fpr = get_rate_vs_k(prf_dists_2_vs_1, prf_train_dists_1_vs_1, k_arr, mean_prf_1)
		prf_2_tpr = get_rate_vs_k(prf_dists_2_vs_2, prf_train_dists_2_vs_2, k_arr, mean_prf_2)
		prf_2_fpr = get_rate_vs_k(prf_dists_1_vs_2, prf_train_dists_2_vs_2, k_arr, mean_prf_2)

		
		prf_data_1 = [prf_1_fpr, prf_1_tpr]
		prf_data_2 = [prf_2_fpr, prf_2_tpr]
		## find points evenly spaced in k here


		if save_samps:
			dist_fname = 'output/ROC/dists/{}_vs_{}.png'
			plot_dists(prf_dists_1_vs_1, prf_train_dists_1_vs_1, var_prf_1, dist_fname.format(label_1, label_1))
			plot_dists(prf_dists_1_vs_2, prf_train_dists_2_vs_2, var_prf_2, dist_fname.format(label_1, label_2))
			plot_dists(prf_dists_2_vs_1, prf_train_dists_1_vs_1, var_prf_1, dist_fname.format(label_2, label_1))
			plot_dists(prf_dists_2_vs_2, prf_train_dists_2_vs_2, var_prf_2, dist_fname.format(label_2, label_2))


		# SPEC STUFF #
		
		spec_dists_1_vs_1 = get_dists(mean_spec_1, specs_test_1)
		spec_dists_2_vs_1 = get_dists(mean_spec_1, specs_test_2)
		spec_dists_1_vs_2 = get_dists(mean_spec_2, specs_test_1)
		spec_dists_2_vs_2 = get_dists(mean_spec_2, specs_test_2)


		spec_train_dists_1_vs_1 = get_dists(mean_spec_1, specs_train_1)
		spec_train_dists_2_vs_2 = get_dists(mean_spec_2, specs_train_2)

		spec_1_tpr = get_rate_vs_k(spec_dists_1_vs_1, spec_train_dists_1_vs_1, k_arr, mean_spec_1)
		spec_1_fpr = get_rate_vs_k(spec_dists_2_vs_1, spec_train_dists_1_vs_1, k_arr, mean_spec_1)
		spec_2_tpr = get_rate_vs_k(spec_dists_2_vs_2, spec_train_dists_2_vs_2, k_arr, mean_spec_2)
		spec_2_fpr = get_rate_vs_k(spec_dists_1_vs_2, spec_train_dists_2_vs_2, k_arr, mean_spec_2)

		spec_data_1 = [spec_1_fpr, spec_1_tpr]
		spec_data_2 = [spec_2_fpr, spec_2_tpr]


		data.append((spec_data_1, spec_data_2, prf_data_1, prf_data_2))

		print_data_spec.append((mean_spec_1, var_spec_1, label_1, mean_spec_2, var_spec_2, label_2))
		print_data_prf.append((mean_prf_1, var_prf_1, label_1, mean_prf_2, var_prf_2, label_2))

	print 'plotting...'
	plot_fig(data, k, label_1, label_2, window_length, out_filename, pub)

	print_stats_multi(print_data_spec, print_data_prf, window_length)

	print 'time elapsed:', time.time() - start

