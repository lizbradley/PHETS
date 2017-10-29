import cPickle

import numpy as np

from PRFStats.plots import dists_vs_means_fig
from Utilities import clear_old_files
from data import L2Classifier
from plots import dual_roc_fig, samples


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


def L2ROCs(
		traj1, traj2,
		label1, label2,
		out_fname,
		filt_params,
		k,
		load_saved=False,
		see_samples=0,
		quiet=True,
		vary_param=None
):

	if load_saved:
		filts1 = cPickle.load(open('PRFStats/data/filts1.p'))
		filts2 = cPickle.load(open('PRFStats/data/filts2.p'))

	else:
		filts1 = []
		filts2 = []

		iterator = 1 if vary_param is None else len(vary_param[1])
		for i in range(iterator):
			if vary_param is not None:
				filt_params.update({vary_param[0]: vary_param[1][i]})

			filts1.append(traj1.filtrations(filt_params, quiet))
			filts2.append(traj2.filtrations(filt_params, quiet))

		cPickle.dump(filts1, open('PRFStats/data/filts1.p', 'wb'))
		cPickle.dump(filts2, open('PRFStats/data/filts2.p', 'wb'))

	data = []

	for f1, f2 in zip(filts1, filts2):

		prfs1 = [f.get_PRF(silent=quiet, new_format=True) for f in f1]
		prfs2 = [f.get_PRF(silent=quiet, new_format=True) for f in f2]

		train1, test1 = prfs1[1::2], prfs1[::2]
		train2, test2 = prfs2[1::2], prfs2[::2]

		print 'training classifiers...'
		clf1 = L2Classifier(train1)
		clf2 = L2Classifier(train2)

		print 'running tests...'
		k_arr = np.arange(*k)
		roc1 = roc_data(clf1, test1, test2, k_arr)
		roc2 = roc_data(clf2, test2, test1, k_arr)

		data.append([roc1, roc2])

	dual_roc_fig(data, k, label1, label2, out_fname, vary_param)

	if see_samples:
		dir = 'output/PRFStats/samples'
		clear_old_files(dir, see_samples)

		if vary_param is None:
			filts1, filts2 = filts1[0], filts2[0]

		samples(filts1, see_samples, dir, vary_param)
		samples(filts2, see_samples, dir, vary_param)

	return data




def plot_dists_vs_means(*args, **kwargs):		# see dists_compare for arg format

	filename_1, filename_2, out_filename, filt_params = args

	sigs_full, crops, sigs, refs, dists = dists_compare(*args, **kwargs)

	dists_vs_means_fig(kwargs, args, sigs_full, crops, sigs, dists)

	base_filename_1 = filename_1.split('/')[-1].split('.')[0]
	base_filename_2 = filename_2.split('/')[-1].split('.')[0]
	out_fname_1 = 'output/PRFCompare/mean/' + base_filename_1 + '_mean_PRF.png'
	out_fname_2 = 'output/PRFCompare/mean/' + base_filename_2 + '_mean_PRF.png'
	ref_func_1, ref_func_2 = refs
	make_PRF_plot(ref_func_1, out_fname_1, params=filt_params,
				  in_filename='MEAN: ' + base_filename_1)
	make_PRF_plot(ref_func_2, out_fname_2, params=filt_params,
				  in_filename='MEAN: ' + base_filename_2)

