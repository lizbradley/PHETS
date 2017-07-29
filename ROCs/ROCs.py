import matplotlib.pyplot as plt
import numpy as np
from PH import Filtration
from PH.Plots import plot_filtration_pub
from config import default_filtration_params as filt_params
from config import WAV_SAMPLE_RATE
from DCE.DCE import embed
from Utilities import print_title
from TimbreDetect.helpers import crop_sig, slice_sig, get_spec, downsample_spec, plot_sig, plot_roc



fname_1 = 'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt'
fname_2 = 'datasets/time_series/viol/40-viol.txt'

label_1 = 'clarinet'
label_2 = 'viol'

crop_length = 70000
c1 = 100000
c2 = 50000

crop_1 = (c1, c1 + crop_length)
crop_2 = (c2, c2 + crop_length)

tau = 32		# samples
m = 2

# window_length = 2000
window_length = (600, 800, 1000)
num_windows = 15
num_landmarks = 50

FT_bins = 50

filt_params.update(
	{
		'max_filtration_param': -10,
		'num_divisions': 10,
		'use_cliques': True,
	}
)

k_min = 0
k_max = 50
k_int = .01
k = np.arange(k_min, k_max, k_int)

load_saved_filts = False



def norm(f):
	if len(f.shape) == 1:
		res = len(f)
	elif len(f.shape) == 2:
		res = len(f) ** .5
	else:
		res = None
		print 'ERROR'


	dA = 2. / (res ** 2)		# normalize such that area of PRF domain is 1
	# dA = 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)


def test_inclusion(mean, var, tests, k):
	var_norm = norm(var)
	pred = []
	for spec in tests:
		diff = np.subtract(spec, mean)
		dist = norm(diff)
		if dist < 0: print 'aaack dist < 0'
		pred.append(dist < var_norm * k)
	return pred



def get_specs(windows):
	specs = [get_spec(w) for w in windows]
	# plot_spec(specs_1[0], 'spec.png')
	specs = np.asarray([downsample_spec(s[0], s[1], FT_bins)[1] for s in specs])
	# plot_spec(specs_ds_1[0], 'spec_ds.png')
	return specs

from PH import make_movie, make_PRF_plot
import datetime

def get_prfs(windows, filt_params, window_length, num_landmarks, label='none', load_saved=False):

	fname = 'ROCs/data/{}_{}.npy'.format(label, window_length)

	if load_saved:
		return np.load(fname)

	trajs = [embed(w, tau, m) for w in windows]
	res = filt_params['num_divisions']
	filts = []

	for i, t in enumerate(trajs):
		print_title('window_length: {} \t label: {} \t window #: {}'.format(
			window_length, label, i))

		filt_params.update({'worm_length': window_length})
		filt_params.update({'ds_rate': window_length / num_landmarks})
		filt = Filtration(t, filt_params)

		fname_base = 'ROCs/samps/wl_{}__num_{}_{}__'.format(
			window_length, i, label)

		# make_movie(filt, 'ROCs/samps/wl_{}__num_{}_{}.mp4'.format(window_length, i, label))
		plot_filtration_pub(filt, 3, fname_base + 'filt.png')
		make_PRF_plot(filt, fname_base + 'PRF.png', PRF_res=res)

		filts.append(filt)

	prfs = [f.get_PRF(res) for f in filts]
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
	return 'is it a {}? [{}])'.format(label, mode)


def plot_fig(data, k):
	fig = plt.figure(figsize=(12, 10))

	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)

	lines = []

	for data_wl in data:
		fft_data_1, fft_data_2, prf_data_1, prf_data_2 = data_wl

		plot_roc(ax1, fft_data_1, roc_title(label_1, 'FFT'))
		plot_roc(ax2, fft_data_2, roc_title(label_2, 'FFT'))
		plot_roc(ax3, prf_data_1, roc_title(label_1, 'PRF'))
		lines.append(plot_roc(ax4, prf_data_2, roc_title(label_2, 'PRF')))

	fig.legend(lines, labels=window_length)
	fig.suptitle('k = range({}, {}, {})'.format(*k), fontsize=16)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	plt.savefig('ROCs/ROCs.png')


def print_stats(mean_1, var_1, label_1, mean_2, var_2, label_2):
	print 'mean {:10}\t{:06.3f}\t\t\tmean {:10}\t{:06.3f}'.format(
		label_1, norm(mean_1), label_2, norm(mean_2))
	print 'var  {:10}\t{:06.3f}\t\t\tvar {:10}\t{:06.3f}'.format(
		label_1, norm(var_1), label_2, norm(var_2))


def print_stats_multi(spec_data, prf_data, window_length):
	for wl, d1, d2 in zip(window_length, spec_data, prf_data):
		print_title(wl)
		print_title('FFT:')
		print_stats(*d1)
		print_title('PRF:')
		print_stats(*d2)



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

for wl in window_length:

	windows_1 = slice_sig(sig_1, l=wl, n=num_windows)
	windows_2 = slice_sig(sig_2, l=wl, n=num_windows)
	
	# plot_sig(sig_1, sig_full_1, windows_1, 'ROCs/sig1.png')
	# plot_sig(sig_2, sig_full_2, windows_2, 'ROCs/sig2.png')

	specs_1 = get_specs(windows_1)
	specs_2 = get_specs(windows_2)

	prfs_1 = get_prfs(windows_1, filt_params, wl, num_landmarks, load_saved=load_saved_filts, label=label_1)
	prfs_2 = get_prfs(windows_2, filt_params, wl, num_landmarks, load_saved=load_saved_filts, label=label_2)

	specs_train_1, specs_test_1, mean_spec_1, var_spec_1 = pre_proc_data(specs_1)
	specs_train_2, specs_test_2, mean_spec_2, var_spec_2 = pre_proc_data(specs_2)

	prfs_train_1, prfs_test_1, mean_prf_1, var_prf_1 = pre_proc_data(prfs_1)
	prfs_train_2, prfs_test_2, mean_prf_2, var_prf_2 = pre_proc_data(prfs_2)


	fft_data_1 = roc_data(mean_spec_1, var_spec_1, specs_test_1, specs_test_2, k)
	fft_data_2 = roc_data(mean_spec_2, var_spec_2, specs_test_2, specs_test_1, k)

	prf_data_1 = roc_data(mean_prf_1, var_prf_1, prfs_test_1, prfs_test_2, k)
	prf_data_2 = roc_data(mean_prf_2, var_prf_2, prfs_test_2, prfs_test_1, k)


	data.append((fft_data_1, fft_data_2, prf_data_1, prf_data_2))

	print_data_spec.append((mean_spec_1, var_spec_1, label_1, mean_spec_2, var_spec_2, label_2))
	print_data_prf.append((mean_prf_1, var_prf_1, label_1, mean_prf_2, var_prf_2, label_2))



k = k_min, k_max, k_int

print 'plotting...'
plot_fig(data, k)

print_stats_multi(print_data_spec, print_data_prf, window_length)

