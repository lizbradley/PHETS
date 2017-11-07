import cPickle

import numpy as np
import sys

def fetch_filts(
		traj1, traj2, params, load_saved, quiet,
		vary_param_1=None, vary_param_2=None
):
	"""
	loads from file, or generates and saves to file
	if no vary params, return filtration evo
	if vary_param_1, not vary_param_2, returns 1d array of filt evos
	if both vary params, return 2d array of filt evos
	"""

	if load_saved:
		filts1 = cPickle.load(open('PRFstats/data/filts1.p'))
		filts2 = cPickle.load(open('PRFstats/data/filts2.p'))
		return filts1, filts2

	iter_1 = 1 if vary_param_1 is None else len(vary_param_1[1])
	iter_2 = 1 if vary_param_2 is None else len(vary_param_2[1])

	filts1_vv, filts2_vv = [], []     # filts varied over two params
	for i in range(iter_1):
		if vary_param_1 is not None:
			params.update({vary_param_1[0]: vary_param_1[1][i]})

		filts1_v, filts2_v = [], []   # filts varied over one param
		for j in range(iter_2):
			if vary_param_2 is not None:
				params.update({vary_param_2[0]: vary_param_2[1][j]})

			filts1_v.append(traj1.filtrations(params, quiet))
			filts2_v.append(traj2.filtrations(params, quiet))

		filts1_vv.append(filts1_v)
		filts2_vv.append(filts2_v)

	filts1_vv, filts2_vv = np.array(filts1_vv), np.array(filts2_vv)

	if vary_param_1 is None and vary_param_2 is not None:
		print 'ERROR: vary_param_1 is None, vary_param_1 is not None'
		sys.exit()
	if vary_param_1 is None and vary_param_2 is None:
		filts1, filts2 = filts1_vv[0, 0], filts2_vv[0, 0]
	elif vary_param_1 is not None and vary_param_2 is None:
		filts1, filts2 = filts1_vv[:, 0], filts2_vv[:, 0]
	else:       # neither are None
		filts1, filts2 = filts1_vv, filts2_vv

	cPickle.dump(filts1, open('PRFstats/data/filts1.p', 'wb'))
	cPickle.dump(filts2, open('PRFstats/data/filts2.p', 'wb'))

	return filts1, filts2

# def norm(f):
# 	dA = 2. / (len(f) ** 2)		# normalize such that area of PRF domain is 1
# 	return np.sqrt(np.nansum(np.power(f, 2)) * dA)


def norm(f, metric='L2'):
	prf_res = len(f)
	dA = 2. / (prf_res ** 2)	   # normalize such that area of PRF domain is 1
	if metric == 'L1':
		return np.nansum(np.abs(f)) * dA
	elif metric == 'L2':
		return np.sqrt(np.nansum(np.power(f, 2)) * dA)
	else:
		print "ERROR: metric not recognized. Use 'L1' or 'L2'."
		sys.exit()


def get_dist(a, b):
	return norm(np.subtract(a, b))





def scale_dists(dists, norms, norm_ref, scale):
	""" helper for get_dists_from_ref """
	if scale == 'none':
		return dists
	elif scale == 'a':
		return np.true_divide(dists, norms)
	elif scale == 'b':
		return np.true_divide(dists, norm_ref)
	elif scale == 'a + b':
		return np.true_divide(dists, np.add(norms, norm_ref))
	else:
		print "ERROR: dist_scale '" + scale + \
		      "' is not recognized. Use 'none', 'a', 'b', or 'a + b'."
		sys.exit()


def get_dists_from_ref(funcs, ref_func, metric, scale):
	dists = [norm(np.subtract(f, ref_func), metric) for f in funcs]
	norms = [norm(f, metric) for f in funcs]
	norm_ref = [norm(ref_func, metric)] * len(dists)
	return scale_dists(dists, norms, norm_ref, scale)


def prf_dists_compare(prfs1, prfs2, metric, dist_scale):
	'''generates and processes data for plot_dists_vs_ref, plot_dists_vs_means,
	 and plot_clusters'''

	mean1 = np.mean(prfs1, axis=0)
	mean2 = np.mean(prfs2, axis=0)

	dists_1_vs_1 = get_dists_from_ref(prfs1, mean1, metric, dist_scale)
	dists_2_vs_1 = get_dists_from_ref(prfs2, mean1, metric, dist_scale)
	dists_1_vs_2 = get_dists_from_ref(prfs1, mean2, metric, dist_scale)
	dists_2_vs_2 = get_dists_from_ref(prfs2, mean2, metric, dist_scale)


	# for plotting ref PRFs #


	arr = [
		[mean1, mean2],
		[dists_1_vs_1, dists_2_vs_1, dists_1_vs_2, dists_2_vs_2]
	]

	return arr


class L2Classifier(object):
	# TODO: add back in dist_scale -- consult nikki
	def __init__(self, train):		# training data as ndarray
		"""
		classifier which compares the l2 distance from the mean of training
		prfs to the test prf, vs the variance of training prfs
		"""
		prfs = train

		self.mean = np.mean(prfs, axis=0)
		self.var = np.var(prfs, axis=0)                # local

		self.dists = [get_dist(self.mean, prf) for prf in prfs]
		# self.gvar = np.mean(self.dists)                     # global
		self.gstddev = np.mean(np.power(self.dists, 2)) ** .5

		self.test_dists = []


	def predict(self, tests, k):

		var_norm = norm(self.var)
		dists = [get_dist(prf, self.mean) for prf in tests]

		# return [dist <= var_norm * k for dist in dists]
		# return [dist <= self.gvar * k for dist in dists]
		return [dist <= self.gstddev * k for dist in dists]


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


