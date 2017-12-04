import os
import numpy as np

from data import dists_to_ref, filt_set, distance, prf_set, NormalPRF
from statscurves.data import pointwise_stats, scaler_stats
from plots import dists_to_means_fig, clusters_fig, dists_to_ref_fig
from statscurves.plots import variance_fig, heatmaps_figs
from PRFstats.plots import weight_functions_figs
from data import mean_dists_compare
from classify.data import DistanceClassifier, roc_data
from plots import samples
from classify.plots import dual_roc_fig
from signals import Trajectory
from utilities import clear_old_files



def plot_dists_to_ref(
		path,
		out_filename,
		filt_params,
		i_ref=15,
		i_arr=np.arange(10, 20, 1),
		weight_func=lambda i, j: 1,
		load_saved_filts=False,
		see_samples=False,
		quiet=True,
		save_filts=True

):
	"""
	plots distance from reference prf over a range of trajectory input files
	"""

	from PH import Filtration
	from utilities import print_title


	if load_saved_filts:
		try:
			saved = np.load(load_saved_filts)
		except AttributeError:
			saved = np.load('PRFstats/data/filts.npy')
		ref_filt, filts = saved

	else:
		filts = []
		for i in i_arr:
			fname = path.format(i)
			print_title(fname)
			traj = Trajectory(fname)
			filts.append(
				Filtration(traj, filt_params, silent=quiet, save=False)
			)
		ref_traj = Trajectory(path.format(i_ref))
		ref_filt = Filtration(ref_traj, filt_params, silent=quiet, save=False)
		if save_filts:
			try:
				np.save(save_filts, [ref_filt, filts])
			except AttributeError:
				np.save('PRFstats/data/filts.npy', [ref_filt, filts])
	filts = np.array(filts)

	prfs = [NormalPRF(f.PRF()) for f in filts]
	ref_prf = NormalPRF(ref_filt.PRF())
	[prf.set_weight(weight_func) for prf in prfs + [ref_prf]]

	dists = dists_to_ref(prfs, ref_prf)
	base_filename = path.split('/')[-1]
	dists_to_ref_fig(base_filename, i_ref, i_arr, dists, out_filename)

	if see_samples:
		dir_ = 'output/PRFstats/samples'
		clear_old_files(dir_, see_samples)
		samples(filts, see_samples, dir_)

	return dists


def plot_dists_to_means(
		traj1,
		traj2,
		out_filename,
		filt_params,
		weight_func=lambda i, j: 1,
		see_samples=False,
		load_saved_filts=False,
		quiet=True
	):

	filts1 = filt_set(
		traj1,
		filt_params,
		load_saved=load_saved_filts[0],
		quiet=quiet,
	    fid=1
	)
	filts2 = filt_set(
		traj2,
		filt_params,
		load_saved=load_saved_filts[1],
		quiet=quiet,
	    fid=2
	)

	prfs1 = prf_set(filts1, weight_func)
	prfs2 = prf_set(filts2, weight_func)

	refs, dists = mean_dists_compare(prfs1, prfs2)

	dists_to_means_fig(refs, dists, traj1, traj2, out_filename)

	if see_samples:
		dir = 'output/PRFstats/samples'
		clear_old_files(dir, see_samples)
		samples(filts1, see_samples, dir)
		samples(filts2, see_samples, dir)

	return dists



def plot_clusters(
		traj1,
		traj2,
		out_filename,
		filt_params,
		weight_func=lambda i, j: 1,
		see_samples=False,
		load_saved_filts=False,
		filts_fnames=(None, None),
		quiet=True
):
	filts1 = filt_set(
		traj1,
		filt_params,
		load_saved=load_saved_filts[0],
		quiet=quiet,
		fid=1
	)
	filts2 = filt_set(
		traj2,
		filt_params,
		load_saved=load_saved_filts[1],
		quiet=quiet,
		fid=2
	)

	prfs1 = prf_set(filts1, weight_func)
	prfs2 = prf_set(filts2, weight_func)

	refs, dists = mean_dists_compare(prfs1, prfs2)

	clusters_fig(dists, filt_params, traj1.name, traj2.name, out_filename)

	if see_samples:
		dir_ = 'output/PRFstats/samples'
		clear_old_files(dir_, see_samples)
		samples(filts1, see_samples, dir_)
		samples(filts2, see_samples, dir_)


def plot_ROCs(
		traj1, traj2,
		out_fname,
		filt_params,
		k,
		weight_func=lambda i, j: 1,
		load_saved_filts=False,
		see_samples=0,
		quiet=True,
		vary_param=None     # ('param', (100, 150, 200))
):
	# TODO: weight function, vary_param_2

	filts1 = filt_set(
		traj1,
		filt_params,
		vary_param,
		load_saved=load_saved_filts[0],
		quiet=quiet,
	    fid=1
	)
	filts2 = filt_set(
		traj2,
		filt_params,
		vary_param,
		load_saved=load_saved_filts[1],
		quiet=quiet,
		fid=2
	)


	prfs1 = prf_set(filts1, weight_func)
	prfs2 = prf_set(filts2, weight_func)

	data = []

	if vary_param is None: prfs1, prfs2 = [prfs1], [prfs2]
	for prfs1_, prfs2_ in zip(prfs1, prfs2):
		train1, test1 = prfs1_[1::2], prfs1_[::2]
		train2, test2 = prfs2_[1::2], prfs2_[::2]

		print 'training classifiers...'
		clf1 = DistanceClassifier(train1)
		clf2 = DistanceClassifier(train2)

		print 'running tests...'
		k_arr = np.arange(*k)
		roc1 = roc_data(clf1, test1, test2, k_arr)
		roc2 = roc_data(clf2, test2, test1, k_arr)

		data.append([roc1, roc2])

	dual_roc_fig(data, k, traj1, traj2, out_fname, vary_param)

	if see_samples:
		dir_ = 'output/PRFstats/samples'
		clear_old_files(dir_, see_samples)
		samples(filts1, see_samples, dir_, vary_param)
		samples(filts2, see_samples, dir_, vary_param)

	return data


def plot_variance(
		traj,
		out_filename,
		filt_params,
		vary_param_1,
		vary_param_2=None,
		legend_labels_1=None,       # ('axis', ('tick 1', 'tick2', 'tick3'))
		legend_labels_2=None,       # ('legend 1', 'legend 2', 'legend 3')
		weight_func=lambda i, j: 1,
		see_samples=False,
		quiet=True,
		annot_hm=False,
		load_saved_filts=False,
		heatmaps=True
):
	out_dir, out_fname = os.path.split(out_filename)
	in_dir, in_fname = os.path.split(traj.fname)

	weight_functions_figs(
		vary_param_1,
		vary_param_2,
		legend_labels_1,
		legend_labels_2,
		weight_func,
		filt_params,
		out_dir
	)

	filts = filt_set(
		traj,
		filt_params,
		vary_param_1,
		vary_param_2,
	    load_saved=load_saved_filts,
	    quiet=quiet
	)

	prfs = prf_set(filts, weight_func, vary_param_1, vary_param_2)
	pointw_data = pointwise_stats(prfs, vary_param_1, vary_param_2)
	scaler_data = scaler_stats(prfs, pointw_data, vary_param_1, vary_param_2)

	variance_fig(
		scaler_data,
		filt_params,
		vary_param_1,
		vary_param_2,
		out_filename,
		legend_labels_1,
		legend_labels_2,
		traj.fname
	)
	if heatmaps:
		heatmaps_figs(
			pointw_data,
			vary_param_1,
			vary_param_2,
			legend_labels_1,
			legend_labels_2,
			out_dir,
			in_fname,
			annot_hm
		)

	if see_samples:
		dir_ = 'output/PRFstats/samples'
		clear_old_files(dir_, see_samples)
		samples(
			filts,
			see_samples,
			'output/PRFstats/samples',
			vary_param_1,
			vary_param_2
		)

	return scaler_data


def pairwise_mean_dists(
		traj,
		filt_params,
		vary_param_1,
		vary_param_2,
		weight_func=lambda i, j: 1,
		see_samples=False,
		quiet=True,
		load_saved_filts=False,
):
	filts = filt_set(
		traj,
		filt_params,
		vary_param_1,
		vary_param_2,
		load_saved=load_saved_filts,
		quiet=quiet,
	)

	prfs = prf_set(filts, weight_func, vary_param_1, vary_param_2)

	means = np.empty(prfs.shape[:-1], dtype=object)

	for idx in np.ndindex(*means.shape):
		mean = NormalPRF.mean(prfs[idx])
		means[idx] = mean

	dists = np.zeros((means.shape[0] - 1, means.shape[1]))
	for i in range(means.shape[0] - 1):
		for j in range(means.shape[1]):
			d = distance(means[i, j], means[i + 1, j])
			dists[i, j] = d

	if see_samples:
		dir_ = 'output/PRFstats/samples'
		clear_old_files(dir_, see_samples)
		samples(
			filts,
			see_samples,
			'output/PRFstats/samples',
			vary_param_1,
			vary_param_2
		)

	return dists


