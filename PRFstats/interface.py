import sys

import numpy as np

from PRFstats.data import roc_data, dists_to_ref, \
	fetch_filts, process_variance_data
from PRFstats.plots import dists_to_means_fig, clusters_fig, dists_to_ref_fig, \
	plot_weight_functions, plot_heatmaps, plot_variane_fig
from data import L2Classifier, mean_dists_compare
from plots import dual_roc_fig, samples
from signals import Trajectory
from utilities import clear_old_files


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
	""" plots distance from reference prf over a range of trajectory input files"""

	# TODO: weight_func

	from PH import Filtration
	import cPickle
	from utilities import print_title

	def win_fname(i):
		if fname_format == 'i base':
			filename = '{}/{}{}'.format(dir, i, base_filename)
		elif fname_format == 'base i':
			filename = '{}/{}{}.txt'.format(dir, base_filename, i)
		else:
			print "ERROR: invalid fname_format. Valid options: 'i base', 'base i'"
			sys.exit()
		return filename

	if load_saved_filts:
		filts = np.load(open('PRFstats/data/filts.p'))
		ref_filt = np.load(open('PRFstats/data/ref_filt.p'))
	else:
		filts = []
		for i in i_arr:
			fname = win_fname(i)
			print_title(fname)
			traj = Trajectory(win_fname(i))
			filts.append(Filtration(traj, filt_params, silent=quiet))
		ref_traj = Trajectory(win_fname(i_ref))
		ref_filt = Filtration(ref_traj, filt_params, silent=quiet)
		cPickle.dump(filts, open('PRFstats/data/filts.p', 'wb'))
		cPickle.dump(ref_filt, open('PRFstats/data/ref_filt.p', 'wb'))


	prfs = [f.PRF(new_format=True) for f in filts]
	ref_prf = ref_filt._PRF(new_format=True)

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
		time_units='samples',
		metric='L2',
		dist_scale='none',              # 'none', 'a', or 'a + b'
		weight_func=lambda i, j: 1,
		see_samples=5,
		load_saved_filts=False,
		filts_fnames=(None, None),
		quiet=True
	):

	# TODO: weight_func, see_samples, time_units, unit test

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

	dists_to_means_fig(refs, dists, traj1, traj2, time_units, out_filename)


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

	# TODO: weight_func, see_samples, unit test


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


def L2ROCs(
		traj1, traj2,
		label1, label2,
		out_fname,
		filt_params,
		k,
		load_saved_filts=False,
		filts_fnames=(None, None),
		see_samples=0,
		quiet=True,
		vary_param=None     # ('param', (100, 150, 200))
):
	# TODO: add weight function

	filts1_v = fetch_filts(
		traj1, filt_params, load_saved_filts, quiet, vary_param,
		id=1, filts_fname=filts_fnames[0]
	)
	filts2_v = fetch_filts(
		traj2, filt_params, load_saved_filts, quiet, vary_param,
		id=2, filts_fname=filts_fnames[1]
	)

	if vary_param is None:
		filts1_v, filts2_v = [filts1_v], [filts2_v]
	data = []

	for filts1, filts2 in zip(filts1_v, filts2_v):

		prfs1 = [f.PRF(silent=quiet, new_format=True) for f in filts1]
		prfs2 = [f.PRF(silent=quiet, new_format=True) for f in filts2]

		train1, test1 = prfs1[1::2], prfs1[::2]
		train2, test2 = prfs2[1::2], prfs2[::2]

		print 'training classifiers...'
		clf1 = L2Classifier(train1)
		clf2 = L2Classifier(train2)

		print 'running tests...'
		k_arr = np.arange(*k)
		roc1 = roc_data(clf1, test1, test2, k_arr)
		roc2 = roc_data(clf2, test2, test1, k_arr)

		data.append([roc1, roc2])

	dual_roc_fig(data, k, label1, label2, out_fname, vary_param)

	if see_samples:
		dir = 'output/PRFstats/samples'
		clear_old_files(dir, see_samples)

		if vary_param is None:
			filts1_v, filts2_v = filts1_v[0], filts2_v[0]

		samples(filts1_v, see_samples, dir, vary_param)
		samples(filts2_v, see_samples, dir, vary_param)

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
	plot_weight_functions(
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

	plot_variane_fig(
		stats_data,
		filt_params,
		vary_param_1,
		vary_param_2,
	    out_filename,
		legend_labels,
		traj.fname
	)

	plot_heatmaps(
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
