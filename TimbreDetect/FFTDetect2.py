import matplotlib.pyplot as plt
import numpy as np

from PH import Filtration
from helpers import plot_spec, plot_spec_x, plot_roc
from config import default_filtration_params as filt_params
from Utilities import print_title

from helpers import crop_sig, slice_sig, get_spec, downsample_spec, plot_sig

fname_1 = 'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt'
fname_2 = 'datasets/time_series/viol/40-viol.txt'

label_1 = 'clarinet'
label_2 = 'viol'

crop_length = 70000
c1 = 50000
c2 = 100000
crop_1 = (c1, c1 + crop_length)
crop_2 = (c2, c2 + crop_length )

window_length = 5000
num_windows = 30

FT_bins = 100

filt_params.update(
	{
		'ds_rate': 50,
		'max_filtration_param': -7,
		'num_divisions': 10,
		'use_cliques': True,
	}
)


k_min = 0
k_max = 1
k_int = .01
k = np.arange(k_min, k_max, k_int)

# print k



def norm(f):
	if len(f.shape) == 1:
		res = len(f)
	elif len(f.shape) == 2:
		# res = len(f) ** 2
		res = len(f) ** .5
	else:
		print 'ERROR'


	res = len(f)

	dA = 2. / (res ** 2)		# normalize such that area of PRF domain is 1
	# dA = 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)


def test_inclusion(mean, var, tests, k):
	var_norm = norm(var)
	pred = []
	for spec in tests:
		diff = np.subtract(spec, mean)
		dist = norm(diff)
		pred.append(dist < var_norm * k)
	return pred



def get_specs(windows):
	specs = [get_spec(w) for w in windows]
	# plot_spec(specs_1[0], 'spec.png')
	specs = np.asarray([downsample_spec(s[0], s[1], FT_bins)[1] for s in specs])
	# plot_spec(specs_ds_1[0], 'spec_ds.png')
	return specs



def get_prfs(windows, filt_params):
	filts = [Filtration(w, filt_params) for w in windows]
	res = filt_params['num_divisions']
	prfs = [f.get_PRF(res) for f in filts]
	prfs = [prf[2] for prf in prfs]
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

	# print 'tpr:', tpr
	# print 'fpr:', fpr
	return [fpr, tpr]



def roc_title(label, mode, k_min, k_max, k_int):
	return 'is it a {}? [{}]\nk = range({}, {}, {})'.format(label, mode, k_min, k_max, k_int)


print 'loading...'
sig_full_1 = np.loadtxt(fname_1)
sig_full_2 = np.loadtxt(fname_2)

print 'processing...'
sig_1 = crop_sig(sig_full_1, crop_1)
sig_2 = crop_sig(sig_full_2, crop_2)


windows_1 = slice_sig(sig_1, l=window_length, n=num_windows)
windows_2 = slice_sig(sig_2, l=window_length, n=num_windows)

# plot_sig(sig_1, sig_full_1, windows_1, label_1)
# plot_sig(sig_2, sig_full_2, windows_2, label_2)


specs_1 = get_specs(windows_1)
specs_2 = get_specs(windows_2)

# plot_spec_x(mean_spec_1, 'mean_spec_1.png')
# plot_spec_x(mean_spec_2, 'mean_spec_2.png')
# plot_spec_x(var_spec_1, 'var_spec_1.png')
# plot_spec_x(var_spec_2, 'var_spec_2.png')
#
# mean_1 = norm(mean_spec_1)
# mean_2 = norm(mean_spec_2)
# var_1 = norm(var_spec_1)
# var_2 = norm(var_spec_2)
#
# print 'norm mean 1: \t {:.3f} \t\t norm mean 2: \t {:.3f}'.format(mean_1, mean_2)
# print 'norm var 1: \t {:.3f} \t norm var 2: \t{:.3f}'.format(var_1, var_2)


specs_train_1, specs_test_1, mean_spec_1, var_spec_1 = pre_proc_data(specs_1)
specs_train_2, specs_test_2, mean_spec_2, var_spec_2 = pre_proc_data(specs_2)

data_1 = roc_data(mean_spec_1, var_spec_1, specs_test_1, specs_test_2, k)
data_2 = roc_data(mean_spec_2, var_spec_2, specs_test_2, specs_test_1, k)

plot_roc(data_1, roc_title(label_1, 'FFT', k_min, k_max, k_int),
		 'clarinet_ROC_FFT.png')

plot_roc(data_2, roc_title(label_2, 'FFT', k_min, k_max, k_int),
		 'viol_ROC_FFT.png')






# prfs_1 = get_prfs(windows_1, filt_params)
# prfs_2 = get_prfs(windows_2, filt_params)
#
# np.save('prfs_1.npy', prfs_1)
# np.save('prfs_2.npy', prfs_2)

prfs_1 = np.load('prfs_1.npy')
prfs_2 = np.load('prfs_2.npy')


prfs_train_1, prfs_test_1, mean_prf_1, var_prf_1 = pre_proc_data(prfs_1)
prfs_train_2, prfs_test_2, mean_prf_2, var_prf_2 = pre_proc_data(prfs_2)

data_1 = roc_data(mean_prf_1, var_prf_1, prfs_test_1, prfs_test_2, k)
data_2 = roc_data(mean_prf_2, var_prf_2, prfs_test_2, prfs_test_1, k)

plot_roc(data_1, roc_title(label_1, 'PRF', k_min, k_max, k_int),
		 'clarinet_ROC_PRF.png' )

plot_roc(data_2, roc_title(label_2, 'PRF', k_min, k_max, k_int),
		 'viol_ROC_PRF.png')




mean_1 = norm(mean_prf_1)
mean_2 = norm(mean_prf_2)
var_1 = norm(var_prf_1)
var_2 = norm(var_prf_2)
print 'PRF:'
print 'norm mean 1: \t {:.3f} \t\t norm mean 2: \t {:.3f}'.format(mean_1, mean_2)
print 'norm var 1: \t {:.3f} \t\t norm var 2: \t{:.3f}'.format(var_1, var_2)



mean_1 = norm(mean_spec_1)
mean_2 = norm(mean_spec_2)
var_1 = norm(var_spec_1)
var_2 = norm(var_spec_2)
print 'FFT'
print 'norm mean 1: \t {:.3f} \t\t norm mean 2: \t {:.3f}'.format(mean_1, mean_2)
print 'norm var 1: \t {:.3f} \t\t norm var 2: \t{:.3f}'.format(var_1, var_2)


