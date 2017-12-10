import copy

import numpy as np

from phomology.data import PRF
from helpers import *
from utilities import timeit


class NormalPRF:
	"""
	A persistence rank function scaled such that the area of its domain is 1.
	This makes the comparison of PRFs with different domains sensible. This is
	especially useful if a set of PRFs has been generated with
	``filt_params['max_filtration_param'] < 0 `` (that is, the filtrations end
	at the epsilon value where their first n-simplex appears.)

	Parameters
	----------
	prf: PRF or 2d array
		source prf

	Class Attributes
	----------------
	dom_area: int or float
		area of normal prf domain
	"""
	dom_area = 1        # area of PRF domain (the triangle)
	lim = np.sqrt(dom_area * 2)

	def __init__(self, prf):
		if isinstance(prf, PRF):
			self.data = prf.data
		elif isinstance(prf, np.ndarray):
			self.data = prf

		self.num_div = self.data.shape[0]
		self.epsilons = np.linspace(0, self.lim, self.num_div)
		self.weight = None
		self.pre_weight = self


	@property
	def norm(self):
		"""
		Returns
		-------
		float
			L2 norm

		"""
		dA = (NormalPRF.lim / self.num_div) ** 2
		return np.sqrt(np.nansum(np.square(self.data)) * dA)


	def set_weight(self, wf):
		"""
		applies weight function to :py:attribute:`data`

		Parameters
		----------
		wf: lambda
			weight function as a lambda function, eg ``lambda i, j: -i + j``
		"""
		self.weight = wf
		self.pre_weight = copy.deepcopy(self)

		x = y = np.linspace(0, np.sqrt(self.dom_area * 2), self.num_div)
		xx, yy = np.meshgrid(x, y)
		wf_arr = wf(xx, yy)
		if isinstance(wf_arr, int):
			wf_arr = np.full_like(xx, wf_arr)
		self.data = np.multiply(self.data, wf_arr)


	@classmethod
	def mean(cls, nprfs):
		"""
		Parameters
		----------
		nprfs: array
			1d array of ``NormalPRF``s

		Returns
		-------
		NormalPRF
			pointwise mean of ``nprfs``

		"""
		raws = [nprf.data for nprf in nprfs]
		return cls(np.mean(raws, axis=0))


	@classmethod
	def var(cls, nprfs):
		"""
		Parameters
		----------
		nprfs: array
			1d array of ``NormalPRF``s

		Returns
		-------
		NormalPRF
			pointwise variance of ``nprfs``

		"""
		raws = [nprf.data for nprf in nprfs]
		return cls(np.var(raws, axis=0))


	@staticmethod
	def _interpret(other):
		if isinstance(other, NormalPRF):
			return other.data
		elif isinstance(other, (int, float, long, np.ndarray)):
			return other
		else:
			raise TypeError

	def __add__(self, other):
		return NormalPRF(self.data + self._interpret(other))

	def __sub__(self, other):
		return NormalPRF(self.data - self._interpret(other))

	def __mul__(self, other):
		return NormalPRF(self.data * self._interpret(other))

	def __div__(self, other):
		return NormalPRF(np.true_divide(self.data,self._interpret(other)))

	def __pow__(self, other):
		return NormalPRF(np.power(self.data, self._interpret(other)))


@timeit
def filt_set(
		traj, params, vp1=None, vp2=None,
        load=False, save=True,
		quiet=True,
        fid=None
) :
	validate_vps(vp1, vp2)

	if load:
		return load_filts(load, fid)

	iter_1 = len(vp1[1]) if is_filt_param(vp1) else 1
	iter_2 = len(vp2[1]) if is_filt_param(vp2) else 1
	filts_vv = []
	for i in range(iter_1):
		if is_filt_param(vp1):
			params.update({vp1[0]: vp1[1][i]})
		filts_v = []
		for j in range(iter_2):
			if is_filt_param(vp2):
				params.update({vp2[0]: vp2[1][j]})
			status_str = status_string(vp1, vp2, i, j)
			filts_ = traj.filtrations(params, quiet, status_str)
			filts_v.append(filts_)
		filts_vv.append(filts_v)
	filts_vv = np.array(filts_vv)
	filts = np.squeeze(filts_vv)

	if save:
		save_filts(save, fid, filts)

	return filts


def prf_set(filts, weight_func=lambda i, j: 1, vp1=None, vp2=None):
	prfs = np.empty_like(filts, dtype=object)
	for idx, filt in np.ndenumerate(filts):
		prfs[idx] = NormalPRF(filt.prf())
		prfs[idx].set_weight(weight_func)
		
	lvp1 = len(vp1[1]) if vp1 is not None else None
	lvp2 = len(vp2[1]) if vp2 is not None else None
	depth = filts.shape[-1]

	def apply_weight_prfs_(prfs_, wf):
		prfs_ = copy.deepcopy(prfs_)
		[prf.set_weight(wf) for prf in prfs_]
		return [prf for prf in prfs_]

	if is_weight_func(vp1):

		if not vp2:
			prfs_ = prfs
			prfs_v = np.empty((lvp1, depth), dtype=object)
			for i, wf in enumerate(vp1[1]):
				prfs_v[i] = apply_weight_prfs_(prfs_, wf)
			prfs = prfs_v

		else:
			prfs_v = prfs
			prfs_vv = np.empty((lvp1, lvp2, depth), dtype=object)
			for i, wf in enumerate(vp1[1]):
				for j, prfs_ in enumerate(prfs_v):
					prfs_vv[i, j] = apply_weight_prfs_(prfs_, wf)
			prfs = prfs_vv

	if is_weight_func(vp2):
		prfs_v = prfs
		prfs_vv = np.empty((lvp1, lvp2, depth), dtype=object)
		for i, prfs_ in enumerate(prfs_v):
			for j, wf in enumerate(vp2[1]):
				prfs_vv[i, j] = apply_weight_prfs_(prfs_, wf)
		prfs = prfs_vv

	return prfs



def distance(a, b):
	return (a - b).norm


def dists_to_ref(funcs, ref_func):
	dists = [(f - ref_func).norm for f in funcs]
	return dists


def mean_dists_compare(prfs1, prfs2):
	"""data for plot_dists_vs_means, and plot_clusters"""

	mean1 = NormalPRF.mean(prfs1)
	mean2 = NormalPRF.mean(prfs2)

	dists_1_vs_1 = dists_to_ref(prfs1, mean1)
	dists_2_vs_1 = dists_to_ref(prfs2, mean1)
	dists_1_vs_2 = dists_to_ref(prfs1, mean2)
	dists_2_vs_2 = dists_to_ref(prfs2, mean2)

	arr = [
		[mean1, mean2],
		[dists_1_vs_1, dists_2_vs_1, dists_1_vs_2, dists_2_vs_2]
	]

	return arr


class L2Classifier(object):
	def __init__(self, train):
		"""
		classifier which compares the distance from the mean of training
		prfs to the test prf, vs the standard deviation of training prfs
		"""
		prfs = train

		self.mean = NormalPRF.mean(prfs)
		self.lvar = NormalPRF.var(prfs)                           # local
		self.lstddev = self.lvar ** .5

		self.dists = [distance(self.mean, prf) for prf in prfs]

		self.gvar = np.mean(np.power(self.dists, 2))               # global
		self.gstddev = self.gvar ** .5

		self.test_dists = []


	def predict(self, test, k, stddev='global'):
		dist = distance(test, self.mean)

		if stddev == 'global':
			measure =  self.gstddev
		elif stddev == 'local':
			measure = self.lstddev.norm
		else:
			raise ParamError("Invalid stddev. Use 'local' or 'global'.")

		return dist <= measure * k


def roc_data(clf, tests_true, tests_false, k_arr):
	tpr = []
	fpr = []
	for k in k_arr:
		true_pos = [clf.predict(t, k) for t in tests_true]
		false_pos = [clf.predict(t, k) for t in tests_false]
		true_pos_rate = sum(true_pos) / float(len(true_pos))
		false_pos_rate = sum(false_pos) / float(len(false_pos))
		tpr.append(true_pos_rate)
		fpr.append(false_pos_rate)

	return [fpr, tpr]