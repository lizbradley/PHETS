import numpy as np

from PRFstats.data import roc_data, dists_to_ref, fetch_filts, distance, \
	fetch_prfs, scaler_stats, pointwise_stats
from PRFstats.plots import dists_to_means_fig, clusters_fig, \
	dists_to_ref_fig, weight_functions_figs, heatmaps_figs, variance_fig
from data import DistanceClassifier, mean_dists_compare
from plots import dual_roc_fig, samples
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
		see_samples=5,
		quiet=True

):
	"""
	plots distance from reference prf over a range of trajectory input files
	"""

	# TODO: weight_func, unit test

	from PH import Filtration
	import cPickle
	from utilities import print_title

	if load_saved_filts:
		filts = np.load(open('PRFstats/data/filts.p'))
		ref_filt = np.load(open('PRFstats/data/ref_filt.p'))
	else:
		filts = []
		for i in i_arr:
			fname = path.format(i)
			print_title(fname)
			traj = Trajectory(fname)
			filts.append(Filtration(traj, filt_params, silent=quiet))
		ref_traj = Trajectory(path.format(i_ref))
		ref_filt = Filtration(ref_traj, filt_params, silent=quiet)
		cPickle.dump(filts, open('PRFstats/data/filts.p', 'wb'))
		cPickle.dump(ref_filt, open('PRFstats/data/ref_filt.p', 'wb'))


	prfs = [f.PRF() for f in filts]
	ref_prf = ref_filt.PRF

	dists = dists_to_ref(prfs, ref_prf)
	base_filename = path.split('/')[-1]
	dists_to_ref_fig(base_filename, i_ref, i_arr, dists, out_filename)

	if see_samples:
		dir_ = 'output/PRFstats/samples'
		clear_old_files(dir_, see_samples)
		samples(filts, see_samples, dir_)


def plot_dists_to_means(
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

	filts1 = fetch_filts(
		traj1, filt_params, load_saved_filts, quiet,
		fid=1, filts_fname=filts_fnames[0]
	)
	filts2 = fetch_filts(
		traj2, filt_params, load_saved_filts, quiet,
		fid=2, filts_fname=filts_fnames[1]
	)

	prfs1 = fetch_prfs(filts1, weight_func, quiet=quiet)
	prfs2 = fetch_prfs(filts2, weight_func, quiet=quiet)

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

	filts1 = fetch_filts(
		traj1, filt_params, load_saved_filts, quiet,
		fid=1, filts_fname=filts_fnames[0]
	)
	filts2 = fetch_filts(
		traj2, filt_params, load_saved_filts, quiet,
		fid=2, filts_fname=filts_fnames[1]
	)

	prfs1 = fetch_prfs(filts1, weight_func, quiet=quiet)
	prfs2 = fetch_prfs(filts2, weight_func, quiet=quiet)

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
		filts_fnames=(None, None),
		see_samples=0,
		quiet=True,
		vary_param=None     # ('param', (100, 150, 200))
):
	# TODO: weight function, vary_param_2

	filts1 = fetch_filts(
		traj1, filt_params, load_saved_filts, quiet, vary_param,
		fid=1, filts_fname=filts_fnames[0]
	)
	filts2 = fetch_filts(
		traj2, filt_params, load_saved_filts, quiet, vary_param,
		fid=2, filts_fname=filts_fnames[1]
	)


	prfs1 = fetch_prfs(filts1, weight_func, quiet=quiet)
	prfs2 = fetch_prfs(filts2, weight_func, quiet=quiet)

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
		legend_labels=None,

		weight_func=lambda i, j: 1,

		see_samples=5,
		quiet=True,
		annot_hm=False,
		load_saved_filts=False,
		filts_fname=None,
		unit_test=False
):

	def sqrt_weight_func(x, y):
		return weight_func(x, y) ** .5

	# plot_trajectory(sig)
	weight_functions_figs(
		vary_param_2,
		legend_labels,
		weight_func,
		filt_params,
		unit_test
	)

	filts = fetch_filts(
		traj, filt_params,
		load_saved_filts, quiet,
		vary_param_1, vary_param_2,
		filts_fname=filts_fname
	)

	prfs = fetch_prfs(
		filts,
		weight_func,
		vary_param_1,
		vary_param_2,
		quiet=quiet
	)

	pw_data = pointwise_stats(
		prfs, vary_param_1, vary_param_2
	)
	# pw_data_pre_weight = pointwise_stats(
	# 	prfs_pre_weight, vary_param_1, vary_param_2
	# )

	scaler_data = scaler_stats(prfs, pw_data, vary_param_1, vary_param_2)

	variance_fig(
		scaler_data,
		filt_params,
		vary_param_1,
		vary_param_2,
	    out_filename,
		legend_labels,
		traj.fname
	)

	heatmaps_figs(
		pw_data,
		# pw_data_pre_weight,
		pw_data,             ## FIX ME
		filt_params,
		vary_param_1,
	    vary_param_2,
		legend_labels,
		out_filename,
		annot_hm,
		unit_test
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

	# return scaler_data, pw_data, pw_data_pre_weight
	return scaler_data, pw_data


def plot_pairwise_mean_dists(
		traj,
		out_filename,
		filt_params,
		vary_param_1,
		vary_param_2=None,
		legend_labels=None,

		weight_func=lambda i, j: 1,

		see_samples=5,
		quiet=True,
		annot_hm=False,
		load_saved_filts=False,
		filts_fname=None,
		unit_test=False
):
	filts = fetch_filts(
		traj, filt_params,
		load_saved_filts, quiet,
		vary_param_1, vary_param_2,
		filts_fname=filts_fname
	)

	prfs = fetch_prfs(filts, weight_func, vary_param_1, vary_param_2, quiet)

	prfs_means = np.mean(prfs, axis=2)

	dists_array = np.zeros((prfs_means.shape[0] - 1, prfs_means.shape[1]))
	for i in range(prfs_means.shape[0] - 1):
		for j in range(prfs_means.shape[1]):
			d = distance(prfs_means[i, j], prfs_means[i + 1, j])
			dists_array[i, j] = d

	np.savetxt('output/PRFstats/pairwise_dists.txt', dists_array)



