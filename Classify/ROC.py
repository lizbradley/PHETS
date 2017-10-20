import cPickle
import numpy as np
import sys

from Classify.Plots import plot_dual_roc_fig
from Data import L2MeanPRF
from Signals import TimeSeries, Trajectory

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
		true_pos = clf.predict(tests_true, k)
		false_pos = clf.predict(tests_false, k)
		true_pos_rate = sum(true_pos) / float(len(true_pos))
		false_pos_rate = sum(false_pos) / float(len(false_pos))
		tpr.append(true_pos_rate)
		fpr.append(false_pos_rate)

	return [fpr, tpr]


def L2MeanPRF_ROCs(
		traj1, traj2,
		label1, label2,
		out_fname,
		filt_params,
		k,
		load_saved=False,
		see_samples=True,
		quiet=True,
		vary_param=None
):

	data = []
	iterator = 1
	if vary_param is not None:
		iterator=len(vary_param[1])

	for i in xrange(0,iterator):

		if vary_param is not None:
			filt_params.update({vary_param[0] : vary_param[1][i]})

		if load_saved:
			filts1 = cPickle.load(open('Classify/data/filts1.p'))
			filts2 = cPickle.load(open('Classify/data/filts2.p'))
		else:
			filts1 = traj1.filtrations(filt_params, quiet)
			filts2 = traj2.filtrations(filt_params, quiet)

			cPickle.dump(filts1, open('Classify/data/filts1.p', 'wb'))
			cPickle.dump(filts2, open('Classify/data/filts2.p', 'wb'))


		prfs1 = [f.get_PRF(silent=quiet) for f in filts1]
		prfs2 = [f.get_PRF(silent=quiet) for f in filts2]

		train1, test1 = prfs1[1::2], prfs1[::2]
		train2, test2 = prfs2[1::2], prfs2[::2]

		print 'training classifiers...'
		clf1 = L2MeanPRF(train1)
		clf2 = L2MeanPRF(train2)

		print 'running tests...'
		k_arr = np.arange(*k)
		roc1 = roc_data(clf1, test1, test2, k_arr)
		roc2 = roc_data(clf2, test2, test1, k_arr)

		data.append([roc1,roc2])

	plot_dual_roc_fig(data, k, label1, label2, out_fname,  vary_param)

	return data




