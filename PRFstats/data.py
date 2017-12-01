import warnings
import numpy as np

from config import default_filtration_params as filt_params


class ParamError(Exception):
	def __init__(self, msg):
		Exception.__init__(self, msg)


def validate_vps(vp1, vp2):
	if vp1 is None and vp2 is not None:
		raise ParamError('vary_param_1 is None, vary_param_2 is not None')
	if vp1[0] == vp2[0]:
		raise ParamError('vary_param_1[0] == vary_param_2[0]')


def is_filt_param(vp):
	if vp is None:
		return False
	else:
		return vp[0] in filt_params


def fetch_filts(
		traj, params, load_saved, quiet,
		vary_param_1=None, vary_param_2=None,
		fid=None, filts_fname=None, out_fname=None,
		save=True
):
	suffix = fid if fid is not None else ''
	default_fname = 'PRFstats/data/filts{}.npy'.format(suffix)

	if load_saved:
		fname = default_fname if filts_fname is None else filts_fname
		return np.load(fname)

	iter_1 = len(vary_param_1[1]) if is_filt_param(vary_param_1) else 1
	iter_2 = len(vary_param_2[1]) if is_filt_param(vary_param_2) else 1

	filts_vv = []
	for i in range(iter_1):
		if vary_param_1 is not None:
			params.update({vary_param_1[0]: vary_param_1[1][i]})

		filts_v = []
		for j in range(iter_2):
			if vary_param_2 is not None:
				params.update({vary_param_2[0]: vary_param_2[1][j]})
			filts_ = traj.filtrations(params, quiet)
			filts_v.append(filts_)
		filts_vv.append(filts_v)
	filts_vv = np.array(filts_vv)

	# filts = np.squeeze(filts_vv)
	filts = {
		(False,  False ): filts_vv[0, 0],
		(False,  True  ): filts_vv[0, :],
		(True,   False ): filts_vv[:, 0],
		(True,   True  ): filts_vv
	}[is_filt_param(vary_param_1), is_filt_param(vary_param_2)]



	fname = default_fname if out_fname is None else out_fname
	if save: np.save(fname, filts)
	return filts


def apply_weight(prf, weight_func):
	""" applies weight to _normalized_ prf"""
	z = prf
	x = y = np.linspace(0, 2 ** .5, len(z))
	xx, yy = np.meshgrid(x, y)

	weight_func = weight_func(xx, yy)
	if isinstance(weight_func, int):
		weight_func = np.full_like(xx, weight_func)

	z = np.multiply(z, weight_func)

	return z


def fetch_prfs(
		filts,
		weight_func,
		vary_param_1=None,
		vary_param_2=None,
		quiet=True
	):
	prfs = np.zeros_like(filts)
	for idx, filt in np.ndenumerate(filts):
		filt.silent = quiet
		prf = filt.PRF().data
		prfs[idx] = apply_weight(prf, weight_func)

	if vary_param_1 and vary_param_1[0] == 'weight_func':
		if not vary_param_2:
			prfs_ = prfs
			prfs_v = np.empty(len(vary_param_1[1]))
			prfs_pre_weight_v = np.empty(len(vary_param_1[1]))
			for i, wf in enumerate(vary_param_1[1]):
				prfs_v[i] = [apply_weight(prf, wf) for prf in prfs_]
			prfs = prfs_v

		else:
			prfs_v = prfs
			prfs_vv = np.empty((len(vary_param_1[1]), len(vary_param_2[1])))
			for i, wf in enumerate(vary_param_1[1]):
				for j, prfs_ in prfs_v:
					prfs_vv[i, j] = [apply_weight(prf, wf) for prf in prfs_]
			prfs = prfs_vv

	if vary_param_2 and vary_param_2[0] == 'weight_func':
		prfs_v = prfs
		prfs_vv = np.empty(len(vary_param_1[1]), len(vary_param_2[1]))
		for i, prfs_ in prfs_v:
			for j, wf in enumerate(vary_param_2[1]):
				prfs_vv[i, j] = [apply_weight(prf, wf) for prf in prfs_]
		prfs = prfs_vv

	return np.asarray(prfs)


def norm(f):
	prf_res = len(f)
	dA = 2. / (prf_res ** 2)	  # normalize such that area of PRF domain is 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)


def distance(a, b):
	return norm(np.subtract(a, b))


def dists_to_ref(funcs, ref_func):
	dists = [norm(np.subtract(f, ref_func)) for f in funcs]
	return dists


def mean_dists_compare(prfs1, prfs2):
	"""data for plot_dists_vs_means, and plot_clusters"""

	mean1 = np.mean(prfs1, axis=0)
	mean2 = np.mean(prfs2, axis=0)

	dists_1_vs_1 = dists_to_ref(prfs1, mean1)
	dists_2_vs_1 = dists_to_ref(prfs2, mean1)
	dists_1_vs_2 = dists_to_ref(prfs1, mean2)
	dists_2_vs_2 = dists_to_ref(prfs2, mean2)

	arr = [
		[mean1, mean2],
		[dists_1_vs_1, dists_2_vs_1, dists_1_vs_2, dists_2_vs_2]
	]

	return arr


class PointwiseStats:
	def __init__(self, prfs):
		self.mean = np.mean(prfs, axis=0)
		self.var = np.var(prfs, axis=0)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.cov = self.var / self.mean

class VarianceData:
	pass

class HeatmapData:
	pass


class NormStats:
	def __init__(self, prfs, pw_stats):
		self.mean = norm(pw_stats.mean)
		self.lvar = norm(pw_stats.var)
		self.lfanofactor = norm(pw_stats.var / pw_stats.mean)
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
		pw_mean = np.mean(prfs)
		dists = np.array([norm(prf - pw_mean) for prf in prfs])
		return np.mean(dists ** 2)



def indices(vp1, vp2):
	if vp2 is None:
		lim1 = len(vp1[1])
		idxs = [i for i in range(lim1)]
		shape = lim1

	else:
		lim1, lim2 = len(vp1[1]), len(vp2[1])
		idxs = [(i, j) for i in range(lim1) for j in range(lim2)]
		shape = (lim1, lim2)

	return shape, idxs


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


class DistanceClassifier(object):
	def __init__(self, train):
		"""
		classifier which compares the distance from the mean of training
		prfs to the test prf, vs the standard deviation of training prfs
		"""
		prfs = train

		self.mean = np.mean(prfs, axis=0)
		self.lvar = np.var(prfs, axis=0)                           # local
		self.lstddev = np.power(self.lvar, .5)

		self.dists = [distance(self.mean, prf) for prf in prfs]

		self.gvar = np.mean(np.power(self.dists, 2))               # global
		self.gstddev = self.gvar ** .5

		self.test_dists = []


	def predict(self, test, k, stddev='global'):
		dist = distance(test, self.mean)

		if stddev == 'global':
			measure =  self.gstddev
		elif stddev == 'local':
			measure = norm(self.lstddev)
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
