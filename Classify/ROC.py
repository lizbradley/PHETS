import numpy as np

from Classify.Plots import plot_dual_roc_fig
from Data import L2MeanPRF


def prep_data(samps):
	samps_train = samps[1::2]
	samps_test = samps[::2]
	mean_samp = np.mean(samps_train, axis=0)
	var_samp = np.var(samps_train, axis=0)
	return samps_train, samps_test, mean_samp, var_samp



def roc_data(clf, tests_true, tests_false, k_arr):
	tpr = []
	fpr = []
	for k in k_arr:
		true_pos = [clf.predict(test, k) for test in tests_true]
		false_pos = [clf.predict(test, k) for test in tests_false]
		true_pos_rate = sum(true_pos) / float(len(true_pos))
		false_pos_rate = sum(false_pos) / float(len(false_pos))
		tpr.append(true_pos_rate)
		fpr.append(false_pos_rate)

	return [fpr, tpr]


def L2MeanPRF_ROCs(
		sig1, sig2,
		label1, label2,

		out_fname,

		k,
		filt_params,

		tau=None, m=None,

		load_saved_filts=False,

		see_samples=True
):



	samps1, samps2 = sig1.windows, sig2.windows
	if None in (samps1, samps2):
		print (
			"ERROR: sig1 and sig2 must be windowed before they are passed to"
			"L2MeanPRF_ROCs. Initialize with 'num_windows' kwarg, or call"
			"'slice' method"
		)

	train1, test1 = samps1[1::2], samps1[::2]
	train2, test2 = samps2[1::2], samps2[::2]

	clf1 = L2MeanPRF(train1, filt_params, tau, m)
	clf2 = L2MeanPRF(train2, filt_params, tau, m)

	k_arr = np.arange(*k)
	roc1 = roc_data(clf1, test1, test2, k_arr)
	roc2 = roc_data(clf2, test2, test1, k_arr)

	plot_dual_roc_fig([[roc1, roc2]], k, label1, label2, out_fname)




