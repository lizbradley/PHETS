import time

import numpy as np
import sys

from DCE import embed
from PH import Filtration
from PubPlots import plot_filtration_pub
from Utilities import print_title
from Signals import TimeSeries, Trajectory



def norm(f):
	dA = 2. / (len(f) ** 2)		# normalize such that area of PRF domain is 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)


def get_dist(a, b):
	return norm(np.subtract(a, b))



class L2Classifier(object):

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



