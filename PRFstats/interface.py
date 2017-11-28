import numpy as np

from PRFstats.data import roc_data, dists_to_ref, \
	fetch_filts, process_variance_data, get_dist, fetch_prfs
from PRFstats.plots import dists_to_means_fig, clusters_fig, dists_to_ref_fig, \
	weight_functions_figs, heatmaps_figs, variance_fig
from data import DistanceClassifier, mean_dists_compare
from plots import dual_roc_fig, samples
from signals import Trajectory
from utilities import clear_old_files


def win_fname(fname_format, dir, base_filename, i):
	if fname_format == 'i base':
		filename = '{}/{}{}'.format(dir, i, base_filename)
	elif fname_format == 'base i':
		filename = '{}/{}{}.txt'.format(dir, base_filename, i)
	else:
		msg = "ERROR: invalid fname_format. Valid options: 'i base', 'base i'"
		raise Exception(msg)
	return filename






def plot_dists_to_ref(
		dir, base_filename,
		fname_format,   # 'i base' or 'base i'
		out_filename,
		filt_params,

		i_ref=15,
		i_arr=np.arange(10, 20, 1),
		weight_func=lambda i, j: 1,
		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', or 'a + b'
		load_saved_filts=False,
		see_samples=5,
		quiet=True

):
	"""
	plots distance from reference prf over a range of trajectory input files
	"""

	# TODO: weight_func

	from PH import Filtration
	import cPickle
	from utilities import print_title

	if load_saved_filts:
		filts = np.load(open('PRFstats/data/filts.p'))
		ref_filt = np.load(open('PRFstats/data/ref_filt.p'))
	else:
		filts = []
		for i in i_arr:
			fname = win_fname(fname_format, dir, base_filename, i)
			print_title(fname)
			traj = Trajectory(win_fname(fname_format, dir, base_filename, i))
			filts.append(Filtration(traj, filt_params, silent=quiet))
		ref_traj = Trajectory(win_fname(fname_format, dir, base_filename,
		                                i_ref))
		ref_filt = Filtration(ref_traj, filt_params, silent=quiet)
		cPickle.dump(filts, open('PRFstats/data/filts.p', 'wb'))
		cPickle.dump(ref_filt, open('PRFstats/data/ref_filt.p', 'wb'))


	prfs = [f.PRF(new_format=True) for f in filts]
	ref_prf = ref_filt.PRF(new_format=True)

	dists = dists_to_ref(prfs, ref_prf, metric, dist_scale)
	dists_to_ref_fig(base_filename, i_ref, i_arr, dists, out_filename)

	if see_samples:
		dir = 'output/PRFstats/samples'
		clear_old_files(dir, see_samples)
		samples(filts, see_samples, dir)



def plot_dists_to_means(
		traj1,
		traj2,
		out_filename,
		filt_params,
		metric='L2',
		dist_scale='none',              # 'none', 'a', or 'a + b'
		weight_func=lambda i, j: 1,
		see_samples=0,
		load_saved_filts=False,
		filts_fnames=(None, None),
		quiet=True
	):

	filts1 = fetch_filts(
		traj1, filt_params, load_saved_filts, quiet,
		id=1, filts_fname=filts_fnames[0]
	)
	filts2 = fetch_filts(
		traj2, filt_params, load_saved_filts, quiet,
		id=2, filts_fname=filts_fnames[1]
	)

	prfs1 = fetch_prfs(filts1, weight_func, quiet=quiet)
	prfs2 = fetch_prfs(filts2, weight_func, quiet=quiet)

	refs, dists = mean_dists_compare(prfs1, prfs2, metric, dist_scale)

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
		metric='L2',
		dist_scale='none',              # 'none', 'a', or 'a + b'
		weight_func=lambda i, j: 1,
		see_samples=5,
		load_saved_filts=False,
		filts_fnames=(None, None),
		quiet=True
):

	# TODO: weight_func, unit test


	filts1 = fetch_filts(
		traj1, filt_params, load_saved_filts, quiet,
		id=1, filts_fname=filts_fnames[0]
	)
	filts2 = fetch_filts(
		traj2, filt_params, load_saved_filts, quiet,
		id=2, filts_fname=filts_fnames[1]
	)

	prfs1 = [f.PRF(silent=quiet, new_format=True) for f in filts1]
	prfs2 = [f.PRF(silent=quiet, new_format=True) for f in filts2]

	refs, dists = mean_dists_compare(prfs1, prfs2, metric, dist_scale)

	clusters_fig(dists, filt_params, traj1.name, traj2.name,out_filename)

	if see_samples:
		dir = 'output/PRFstats/samples'
		clear_old_files(dir, see_samples)
		samples(filts1, see_samples, dir)
		samples(filts2, see_samples, dir)


def plot_ROCs(
		traj1, traj2,
		out_fname,
		filt_params,
		k,
		metric='L2',
		dist_scale='none',
		load_saved_filts=False,
		filts_fnames=(None, None),
		see_samples=0,
		quiet=True,
		vary_param=None     # ('param', (100, 150, 200))
):
	# TODO: weight function, vary_param_2

	filts1 = fetch_filts(
		traj1, filt_params, load_saved_filts, quiet, vary_param,
		id=1, filts_fname=filts_fnames[0]
	)
	filts2 = fetch_filts(
		traj2, filt_params, load_saved_filts, quiet, vary_param,
		id=2, filts_fname=filts_fnames[1]
	)

	data = []
	debugprfs1, debugprfs2 = [], []
	if vary_param is None: filts1, filts2 = [filts1], [filts2]
	for filts1, filts2 in zip(filts1, filts2):

		prfs1 = [f.PRF(silent=quiet, new_format=True) for f in filts1]
		prfs2 = [f.PRF(silent=quiet, new_format=True) for f in filts2]

		debugprfs1.append(prfs1)
		debugprfs2.append(prfs2)

		train1, test1 = prfs1[1::2], prfs1[::2]
		train2, test2 = prfs2[1::2], prfs2[::2]

		print 'training classifiers...'
		clf1 = DistanceClassifier(train1, metric, dist_scale)
		clf2 = DistanceClassifier(train2, metric, dist_scale)

		print 'running tests...'
		k_arr = np.arange(*k)
		roc1 = roc_data(clf1, test1, test2, k_arr)
		roc2 = roc_data(clf2, test2, test1, k_arr)

		data.append([roc1, roc2])
	if vary_param is None: filts1, filts2 = filts1[0], filts2[0]

	np.save('debugprfs1_ref.npy', debugprfs1)
	np.save('debugprfs2_ref.npy', debugprfs2)
	dual_roc_fig(data, k, traj1, traj2, out_fname, vary_param)

	if see_samples:
		dir = 'output/PRFstats/samples'
		clear_old_files(dir, see_samples)
		samples(filts1, see_samples, dir, vary_param)
		samples(filts2, see_samples, dir, vary_param)

	return data


def plot_variance(
		traj,
		out_filename,
		filt_params,
		vary_param_1,
		vary_param_2=None,
		legend_labels=None,

		metric='L2', 		 		# 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',
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

	filt_evo_array = fetch_filts(
		traj, filt_params,
		load_saved_filts, quiet,
		vary_param_1, vary_param_2,
		filts_fname=filts_fname
	)


	stats_data, hmap_data, hmap_data_pw = process_variance_data(
		filt_evo_array,
		metric,
		dist_scale,
		sqrt_weight_func,
		vary_param_2
	)

	variance_fig(
		stats_data,
		filt_params,
		vary_param_1,
		vary_param_2,
	    out_filename,
		legend_labels,
		traj.fname
	)

	heatmaps_figs(
		hmap_data,
		hmap_data_pw,
		filt_params,
		vary_param_1,
	    vary_param_2,
		legend_labels,
		out_filename,
		annot_hm,
		unit_test
	)

	if see_samples:
		dir = 'output/PRFstats/samples'
		clear_old_files(dir, see_samples)
		samples(
			filt_evo_array,
			see_samples,
			'output/PRFstats/samples',
			vary_param_1,
			vary_param_2
		)

	return stats_data, hmap_data, hmap_data_pw


def plot_pairwise_mean_dists(
		traj,
		out_filename,
		filt_params,
		vary_param_1,
		vary_param_2=None,
		legend_labels=None,

		metric='L2', 		 		# 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',
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
			d = get_dist(prfs_means[i, j], prfs_means[i + 1, j])
			dists_array[i, j] = d

	np.savetxt('output/PRFstats/pairwise_dists.txt', dists_array)



