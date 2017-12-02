import warnings

import numpy as np

from PRFstats.data import indices, NormalPRF


class PointwiseStats:
	def __init__(self, prfs):
		self.mean = NormalPRF.mean(prfs)
		self.var = NormalPRF.var(prfs)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.fanofactor = self.var / self.mean


class NormStats:
	def __init__(self, prfs, pw_stats):
		self.mean = pw_stats.mean.norm
		self.lvar = pw_stats.var.norm
		self.lfanofactor = pw_stats.fanofactor.norm
		self.lfanofactor2 = self.lvar / self.mean

		self.gvar = self.global_variance(prfs)
		self.gfanofactor = self.gvar / self.mean

		print '''
		lfanofactor: {} 
		lfanofactor2: {}
		gfanofactor: {} 
		'''.format(self.lfanofactor, self.lfanofactor2, self.gfanofactor)

	@staticmethod
	def global_variance(prfs):
		pw_mean = NormalPRF.mean(prfs)
		dists = np.array([(prf - pw_mean).norm for prf in prfs])
		return np.mean(dists ** 2)


def pointwise_stats(prfs, vary_param_1, vary_param_2):
	shape, idxs = indices(vary_param_1, vary_param_2)
	data = np.empty(shape, dtype=object)
	for idx in idxs:
		data[idx] = PointwiseStats(prfs[idx])
	return data


def scaler_stats(prfs, pw_stats, vary_param_1, vary_param_2):
	shape, idxs = indices(vary_param_1, vary_param_2)
	data = np.empty(shape, dtype=object)
	for idx in idxs:
		data[idx] = NormStats(prfs[idx], pw_stats[idx])
	return data