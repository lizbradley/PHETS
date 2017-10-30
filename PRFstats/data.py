import time

import numpy as np
import sys

from DCE import embed
from PH import Filtration
from PubPlots import plot_filtration_pub
from Utilities import print_title
from signals import TimeSeries, Trajectory


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
