import time

import numpy as np

from DCE import embed
from PH import Filtration
from PubPlots import plot_filtration_pub
from Utilities import print_title
from Signals import TimeSeries, Trajectory
from config import default_filtration_params as filt_params



def norm(f):
	dA = 2. / (len(f) ** 2)		# normalize such that area of PRF domain is 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)


from PH import make_PRF_plot
from Utilities import clear_dir



class L2MeanPRF(object):

	def __init__(self,
		train,		# training data as ndarray
        filt_params,
		tau=None,
		m=None,


	):
		"""
		classifier which compares the l2 distance from the mean of training
		prfs to the test prf, vs the variance of training prfs
		"""


		self.trajs = [embed(w, tau, m) for w in train]

		self.filts = [Filtration(t, filt_params) for t in self.trajs]

		self.prfs = [f.get_PRF for f in self.filts]


		self.mean = np.mean(self.prfs, axis=0)
		self.var = np.var(self.prfs, axis=0)                # local

		self.dists = self.get_dists(self.mean, self.prfs)
		self.gvar = np.mean(self.dists)                     # global



	@staticmethod
	def get_dists(mean, tests):
		return [norm(np.subtract(test, mean)) for test in tests]


	def predict(self, prf, k):
		var_norm = norm(self.var)

		diff = np.subtract(prf, self.mean)

		dist = norm(diff)
		return dist <= var_norm * k




