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
		id=None, filts_fname=None, out_fname=None,
		save=True
):
	suffix = id if id is not None else ''
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
		prf = filt.PRF(silent=quiet, new_format=True)
		prfs[idx] = apply_weight(prf, weight_func)

	if vary_param_1 and vary_param_1[0] == 'weight_func':
		if not vary_param_2:
			prfs_ = prfs
			prfs_v = np.empty(len(vary_param_1[1]))
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


def norm(f, metric='L2'):
	prf_res = len(f)
	dA = 2. / (prf_res ** 2)	  # normalize such that area of PRF domain is 1
	if metric == 'L1':
		return np.nansum(np.abs(f)) * dA
	elif metric == 'L2':
		return np.sqrt(np.nansum(np.power(f, 2)) * dA)
	else:
		raise ParamError("Invalid metric. Use 'L1' or 'L2'")


def get_dist(a, b, metric='L2'):
	return norm(np.subtract(a, b), metric)


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
		msg = "Invalid dist_scale. Use 'none', 'a', 'b', or 'a + b'."
		raise ParamError(msg)


def dists_to_ref(funcs, ref_func, metric, scale):
	dists = [norm(np.subtract(f, ref_func), metric) for f in funcs]
	norms = [norm(f, metric) for f in funcs]
	norm_ref = [norm(ref_func, metric)] * len(dists)
	return scale_dists(dists, norms, norm_ref, scale)


def mean_dists_compare(prfs1, prfs2, metric, dist_scale):
	"""generates and processes data for plot_dists_vs_means, and plot_clusters"""

	mean1 = np.mean(prfs1, axis=0)
	mean2 = np.mean(prfs2, axis=0)

	dists_1_vs_1 = dists_to_ref(prfs1, mean1, metric, dist_scale)
	dists_2_vs_1 = dists_to_ref(prfs2, mean1, metric, dist_scale)
	dists_1_vs_2 = dists_to_ref(prfs1, mean2, metric, dist_scale)
	dists_2_vs_2 = dists_to_ref(prfs2, mean2, metric, dist_scale)

	arr = [
		[mean1, mean2],
		[dists_1_vs_1, dists_2_vs_1, dists_1_vs_2, dists_2_vs_2]
	]

	return arr


class VarianceData:
	"""all data for a fixed value of vary_param_2 -- one curve per plot"""
	def __init__(self):
		self.pointwise_mean_norm = []
		self.variance = []
		self.scaled_variance = []
		self.pointwise_variance_norm = []
		self.functional_COV_norm = []

class HeatmapData:

	def __init__(self):
		self.pointwise_mean = [[]]
		self.pointwise_var = [[]]
		self.functional_COV = [[]]



def process_variance_data(
		filt_evo_array,
		metric,
		dist_scale,
		weight_func,
		vary_param_2
):

	def apply_weight_to_evo(prf_evo, weight_f):
		weighted_prf_evo = []
		for prf in prf_evo:
			weighted_prf_evo.append(apply_weight(prf, weight_f))
		return np.asarray(weighted_prf_evo)


	def apply_weight_func_to_array(prf_evo_array, weight_f):
		for row in prf_evo_array:
			for evo in row:
				evo[...] = apply_weight_to_evo(evo, weight_f)
		return prf_evo_array


	def vary_evos_over_weight_func(prf_evos):
		prf_evos_1d = prf_evos[0]
		prf_evos_2d = []
		for prf_evo in prf_evos_1d:
			prf_evo_vary_2 = []
			for weight_f in vary_param_2[1]:
				weighted_prf_evo = apply_weight_to_evo(prf_evo, weight_f)
				prf_evo_vary_2.append(weighted_prf_evo)
			prf_evos_2d.append(prf_evo_vary_2)

		prf_evos_2d = np.transpose(np.asarray(prf_evos_2d), (1, 0, 2, 3))
		return prf_evos_2d


	def calculate_stats(prf_evos_1d, apply_weight_to_fcov=True):
		var_data = VarianceData()
		hmap_data_arr = []

		for prf_evo in prf_evos_1d:  # for each value of vary_param_1

			hmap_data = HeatmapData()
			# see definitions for norm() and get_dists_from_ref() around lines 45 - 90

			pointwise_mean = np.mean(prf_evo, axis=0)  				# plot as heatmap
			hmap_data.pointwise_mean = pointwise_mean

			pmn = norm(pointwise_mean, metric)  					# plot as data point
			var_data.pointwise_mean_norm.append(pmn)

			dists = [norm(np.subtract(PRF, pointwise_mean), metric)
			            for PRF in prf_evo]
			variance = np.mean(np.power(dists, 2))  				# plot as data point
			# variance = np.sum(np.power(dists, 2)) / (len(dists) - 1)
			var_data.variance.append(variance)

			scaled_dists = dists_to_ref(
				prf_evo, pointwise_mean, metric, dist_scale
			)
			scaled_variance = np.mean(np.power(scaled_dists, 2))    # plot as data point
			var_data.scaled_variance.append(scaled_variance)

			diffs = [PRF - pointwise_mean for PRF in prf_evo]

			pointwise_variance = np.var(diffs, axis=0) 				# plot as heatmap
			hmap_data.pointwise_var = pointwise_variance

			pvn = norm(pointwise_variance, metric) 					# plot as data point
			var_data.pointwise_variance_norm.append(pvn)

			import warnings
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				FCOV = pointwise_variance / pointwise_mean  		 # plot as heatmap

			if apply_weight_to_fcov:
				hmap_data.functional_COV = apply_weight(FCOV, weight_func)
			else:
				hmap_data.functional_COV = FCOV

			fcovn = norm(FCOV, metric)  # plot as data point
			var_data.functional_COV_norm.append(fcovn)

			hmap_data_arr.append(hmap_data)

		return var_data, hmap_data_arr



	print 'processing data...'

	if vary_param_2 is None:
		filt_evo_array = np.asarray([filt_evo_array])
	else:
		filt_evo_array = filt_evo_array.transpose((1, 0, 2))

	prf_evo_array = np.asarray(
		[[[f.PRF(new_format=True) for f in evo]
			for evo in row]
	            for row in filt_evo_array]
	)

	prf_evo_array_pre_weight = prf_evo_array

	if vary_param_2 and vary_param_2[0] == 'weight_func':
		prf_evo_array = vary_evos_over_weight_func(prf_evo_array)
	elif vary_param_2:
		prf_evo_array = apply_weight_func_to_array(prf_evo_array, weight_func)

	curve_data = []
	hmap_data = []
	for row in prf_evo_array:
		cd, hmd = calculate_stats(row)
		curve_data.append(cd)
		hmap_data.append(hmd)

	hmap_data_pre_weight = []
	for row in prf_evo_array_pre_weight:
		cd, hmd_pre_weight = calculate_stats(row, apply_weight_to_fcov=False)
		hmap_data_pre_weight.append(hmd_pre_weight)

	curve_data = np.asarray(curve_data)
	hmap_data = np.asarray(hmap_data)
	hmap_data_pre_weight = np.asarray(hmap_data_pre_weight)

	return curve_data, hmap_data, hmap_data_pre_weight


class DistanceClassifier(object):
	def __init__(self, train, metric='L2', dist_scale='none'):
		"""
		classifier which compares the distance from the mean of training
		prfs to the test prf, vs the standard deviation of training prfs
		"""
		prfs = train
		self.metric = metric

		self.mean = np.mean(prfs, axis=0)
		self.lvar = np.var(prfs, axis=0)                           # local
		self.lstddev = np.power(self.lvar, .5)

		self.dists = [get_dist(self.mean, prf) for prf in prfs]

		mean_norm = norm(self.mean, metric)
		norms = [norm(prf, metric) for prf in prfs]
		self.dists = scale_dists(self.dists, mean_norm, norms, dist_scale)

		self.gvar = np.mean(np.power(self.dists, 2))               # global
		self.gstddev = self.gvar ** .5

		self.test_dists = []


	def predict(self, test, k, stddev='global'):
		dist = get_dist(test, self.mean)

		if stddev == 'global':
			measure =  self.gstddev
		elif stddev == 'local':
			measure = norm(self.lstddev, self.metric)
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


