"""
save reference outputs for unit tests
should be run infrequently -- only when behavior is intentionally changed
"""

import os, sys
import numpy as np

from paths import root_dir, current_dir

sys.path.append(root_dir)
os.chdir(current_dir)

from common import *

from PRFstats import *


def ref__filt_set_v():
	filt_set(
		ellipse_traj,
		filt_params,
		vp1=('ds_rate', (5, 7, 9)),
		quiet=False,
		save='ref/filt_set_v.npy'
	)


def ref__plot_dists_to_ref():
	out = plot_dists_to_ref(
		root_dir + '/datasets/trajectories/L63_x_m2/L63_x_m2_tau{}.txt',
		'output/dists_to_ref.png',
		filt_params,
		i_ref=15,
		i_arr=np.arange(2, 30, 7),
		quiet=True,
		load_saved_filts='data/dists_to_ref_filts.npy',
	)
	np.save('ref/dists_to_ref.npy', out)


def ref__plot_dists_to_means():
	out = plot_dists_to_means(
		clar_traj, viol_traj,
		'output/plot_dists_to_means.png',
		filt_params,
		load_saved_filts=('data/clar_filts_.npy', 'data/viol_filts_.npy'),
	)
	np.save('ref/plot_dists_to_means.npy', out)


def ref__plot_ROCs():
	out = plot_ROCs(
		clar_traj, viol_traj,
		'output/ROCs.png',
		filt_params,
		vary_param=('ds_rate', np.arange(100, 150, 10)),
		k=(0, 5.01, .1),
		quiet=False,
		load_saved_filts=('data/clar_filts_v.npy', 'data/viol_filts_v.npy'),
	)
	np.save('ref/plot_ROCs_v.npy', out)


def ref__plot_variance_vw():

	f1 = lambda i, j: .1 * (i + j)
	f2 = lambda i, j: .2 * (i + j)
	f3 = lambda i, j: .3 * (i + j)

	out = plot_variance(
		clar_traj,
		'output/plot_variance_vw.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		vary_param_2=('weight_func', (f1, f2, f3)),
		legend_labels_2=('k=.1', 'k=.2', 'k=.3'),
		load_saved_filts='data/clar_filts_v.npy',
		see_samples=False,
		heatmaps=False
	)
	out = plot_variance__extract_output(out)
	np.save('ref/plot_variance_vw.npy', out)


def ref__plot_variance_vv():

	out = plot_variance(
		viol_traj,
		'output/plot_variance_vv.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=False,
		see_samples=False,
		load_saved_filts='data/viol_filts_vv.npy',
		heatmaps=False
	)

	out = plot_variance__extract_output(out)
	np.save('ref/plot_variance_vv.npy', out)


def ref__plot_variance_wv():
	f1 = lambda i, j: 1 * (-i + j)
	f2 = lambda i, j: 2 * (-i + j)
	f3 = lambda i, j: 3 * (-i + j)
	out = plot_variance(
		clar_traj,
		'output/plot_variance_wv.png',
		filt_params,
		vary_param_1=('weight_func', (f1, f2, f3)),
		vary_param_2=('ds_rate', np.arange(100, 150, 10)),
		legend_labels_1=('weight_function', ('k=1', 'k=2', 'k=3')),
		load_saved_filts='data/clar_filts_v.npy',
		heatmaps=False
	)
	out = plot_variance__extract_output(out)
	np.save('ref/plot_variance_wv.npy', out)


def ref__plot_variance_v():
	out = plot_variance(
		clar_traj,
		'output/plot_variance_v.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		quiet=True,
		load_saved_filts='data/clar_filts_v.npy',
		heatmaps=False
	)
	out = plot_variance__extract_output(out)
	np.save('ref/plot_variance_v.npy', out)


def ref__plot_variance_w():
	f1 = lambda i, j: 1 * (-i + j)
	f2 = lambda i, j: 2 * (-i + j)
	f3 = lambda i, j: 3 * (-i + j)
	out = plot_variance(
		clar_traj,
		'output/PRFstats/plot_variance_w.png',
		filt_params,
		vary_param_1=('weight_func', (f1, f2, f3)),
		legend_labels_1=('weight function', ('k=1', 'k=2', 'k=3')),
		quiet=False,
		load_saved_filts='data/clar_filts_.npy',
		heatmaps=False
	)
	out = plot_variance__extract_output(out)
	np.save('ref/plot_variance_w.npy', out)


def ref__pairwise_mean_dists():
	dists = pairwise_mean_dists(
		clar_traj,
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=True,
		load_saved_filts='data/viol_filts_vv.npy',
	)
	np.save('ref/pairwise_mean_dists.npy', dists)


if __name__ == '__main__':
	# ref__filt_set_v()
	# ref__plot_dists_to_ref()
	# ref__plot_dists_to_means()
	# ref__plot_ROCs()
	# ref__plot_variance_vw()
	# ref__plot_variance_vv()
	# ref__plot_variance_wv()
	# ref__plot_variance_v()
	# ref__plot_variance_w()
	# ref__pairwise_mean_dists()
	pass