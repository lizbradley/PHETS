import numpy as np

from PH.filtration import PRankFunction
from config import default_filtration_params as filt_params


class ParamError(Exception):
	def __init__(self, msg):
		Exception.__init__(self, msg)


class NormalPRF:

	dom_area = 1        # area of PRF domain (the triangle)

	def __init__(self, prf):
		if isinstance(prf, PRankFunction):
			self.data = prf.data
		elif isinstance(prf, np.ndarray):
			self.data = prf

		self.num_div = self.data.shape[0]
		self.weights = []
		self.pre_weight = None

	@property
	def norm(self):
		dA = (NormalPRF.dom_area * 2.) / (self.num_div ** 2)
		return np.sqrt(np.nansum(np.square(self.data)) * dA)

	def apply_weight(self, wf):

		if self.weights:
			self.pre_weight = self.data.copy()

		self.weights.append(wf)

		x = y = np.linspace(0, np.sqrt(self.dom_area * 2), self.num_div)
		xx, yy = np.meshgrid(x, y)
		wf_arr = wf(xx, yy)
		if isinstance(wf_arr, int):
			wf_arr = np.full_like(xx, wf_arr)
		self.data = np.multiply(self.data, wf_arr)

	@classmethod
	def mean(cls, nprfs):
		raws = [nprf.data for nprf in nprfs]
		return cls(np.mean(raws, axis=0))

	@classmethod
	def var(cls, nprfs):
		raws = [nprf.data for nprf in nprfs]
		return cls(np.var(raws, axis=0))

	@staticmethod
	def interpret(other):
		if isinstance(other, NormalPRF):
			return other.data
		elif isinstance(other, (int, float, long, np.ndarray)):
			return other
		else:
			raise TypeError

	def __add__(self, other):
		return NormalPRF(self.data + self.interpret(other))

	def __sub__(self, other):
		return NormalPRF(self.data - self.interpret(other))

	def __mul__(self, other):
		return NormalPRF(self.data * self.interpret(other))

	def __div__(self, other):
		return NormalPRF(self.data / self.interpret(other))

	def __pow__(self, other):
		return NormalPRF(np.power(self.data, self.interpret(other)))


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


def is_weight_func(vp):
	return vp and vp[0] == 'weight_func'


def default_fname(fid):
	suffix = fid if fid is not None else ''
	return 'PRFstats/data/filts{}.npy'.format(suffix)


def load_filts(load_saved, fid):
	try:
		return np.load(load_saved)
	except AttributeError:
		return np.load(default_fname(fid))


def save_filts(save, fid, filts):
	try:
		np.save(save, filts)
	except AttributeError:
		np.save(default_fname(fid), filts)


def filt_set(
		traj, params, vp1=None, vp2=None,
        load_saved=False, save=True,
		quiet=True,
        fid=None
) :
	if load_saved:
		return load_filts(load_saved, fid)

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
			status_str = 'vp1: {}, vp2: {}'.format(vp1[1][i], vp2[1][j])
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
		prfs[idx] = NormalPRF(filt.PRF())
		prfs[idx].apply_weight(weight_func)
		
	lvp1 = len(vp1[1]) if vp1 is not None else None
	lvp2 = len(vp2[1]) if vp2 is not None else None
	depth = filts.shape[-1]

	def apply_weight_prfs_(prfs_, wf):
		return [prf.apply_weight(wf) for prf in prfs_]

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
				for j, prfs_ in prfs_v:
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


def apply_weight(prf, weight_func):
	""" applies weight to _normalized_ prf"""
	z = prf
	x = y = np.linspace(0, 2 ** .5, len(z))
	xx, yy = np.meshgrid(x, y)
	weight_func = weight_func(xx, yy)
	if isinstance(weight_func, int):
		weight_func = np.full_like(xx, weight_func)
	return np.multiply(z, weight_func)


def distance(a, b):
	return (a - b).norm


def dists_to_ref(funcs, ref_func):
	dists = [(f - ref_func).norm for f in funcs]
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




