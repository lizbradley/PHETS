import time

import numpy as np

from DCE import embed
from PH import Filtration
from PubPlots import plot_filtration_pub
from Utilities import print_title
from Signals import TimeSeries, Trajectory
from config import default_filtration_params as filt_params



def norm(f):
	if len(f.shape) == 1:
		res = len(f)
	elif len(f.shape) == 2:
		res = len(f)
	else:
		res = None
		print 'ERROR'

	dA = 2. / (res ** 2)		# normalize such that area of PRF domain is 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)





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






class L2MeanPRF(object):

	def __init__(self,
		train,		# training data as ndarray
		label,
        filt_params,
		tau=None,
		m=None,


	):
		""" classifier which calculates and compares l2 distance to a mean prf """

		self.trajs = [embed(w, tau, m) for w in train.windows()]

		self.filts = [Filtration(t, filt_params) for t in self.trajs]

		self.prfs = [f.get_PRF for f in self.filts]


		self.mean = np.mean(self.prfs, axis=0)
		self.var = np.var(self.prfs, axis=0)                # local

		self.dists = self.get_dists(self.mean, self.prfs)
		self.gvar = np.mean(self.dists)                     # global



	@staticmethod
	def get_dists(mean, tests):
		return [norm(np.subtract(test, mean)) for test in tests]


	def test_inclusion(self, prf, k):
		var_norm = norm(self.var)

		pred = []
		diff = np.subtract(prf, self.mean)

		dist = norm(diff)
		pred.append(dist <= var_norm * k)
		return pred




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
		true_pos = clf.test_inclusion(tests_true, k)
		false_pos = clf.test_inclusion(tests_false, k)
		true_pos_rate = sum(true_pos) / float(len(true_pos))
		false_pos_rate = sum(false_pos) / float(len(false_pos))
		tpr.append(true_pos_rate)
		fpr.append(false_pos_rate)

	return [fpr, tpr]



