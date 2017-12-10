import cPickle
import numpy as np

from paths import chdir; chdir()
from paths import root_dir
from prfstats import *
from common import *


def test__filt_set_v():
	chdir()
	fp = filt_params.copy()
	out = filt_set(
		ellipse_traj,
		fp,
	    vp1=('ds_rate', (5, 7, 9)),
		save=False
	)
	out = filt_set_extract_output(out)
	ref = np.load('ref/filt_set_v.npy')

	np.testing.assert_array_equal(out, ref)


def test__plot_dists_to_ref():
	chdir()
	fp = filt_params.copy()
	fp.update({
		'max_filtration_param': -10,
		'num_divisions': 10,
		'ds_rate': 500
	})

	out = plot_dists_to_ref(
		root_dir + '/datasets/trajectories/L63_x_m2/L63_x_m2_tau{}.txt',
		'output/dists_to_ref.png',
		fp,
		i_ref=15,
		i_arr=np.arange(2, 30, 7),
		quiet=True,
		load_saved_filts='data/dists_to_ref_filts.npy',
	)
	ref = np.load('ref/dists_to_ref.npy')
	np.testing.assert_array_equal(out, ref)


def test__plot_dists_to_means():
	chdir()
	out = plot_dists_to_means(
		clar_traj, viol_traj,
		'output/dists_to_means.png',
		filt_params,
		load_saved_filts=('data/clar_filts_.npy', 'data/viol_filts_.npy'),
	)
	ref = np.load('ref/plot_dists_to_means.npy')
	np.testing.assert_array_equal(out, ref)


def test__plot_ROCs_v():
	chdir()
	out = plot_l2rocs(
		clar_traj, viol_traj,
		'output/rocs.png',
		filt_params,
		vary_param=('ds_rate', np.arange(100, 150, 10)),
		k=(0, 5.01, .1),
		quiet=True,
		load_saved_filts=('data/clar_filts_v.npy', 'data/viol_filts_v.npy'),
	)
	ref = np.load('ref/plot_ROCs_v.npy')
	np.testing.assert_array_equal(out, ref)


def test__plot_variance_vw():
	chdir()

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
		quiet=True,
		load_saved_filts='data/clar_filts_v.npy',
		see_samples=False,
		heatmaps=False
	)
	out = plot_variance__extract_output(out)
	ref = np.load('ref/plot_variance_vw.npy')
	np.testing.assert_array_equal(out, ref)


def test__plot_variance_vv():
	chdir()
	fp = filt_params.copy()
	out = plot_variance(
		viol_traj,
		'output/plot_variance_vv.png',
		fp,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=True,
		load_saved_filts='data/viol_filts_vv.npy',
		see_samples=False,
		heatmaps=False
	)
	out = plot_variance__extract_output(out)
	ref = np.load('ref/plot_variance_vv.npy')
	np.testing.assert_array_equal(out, ref)


def test__plot_variance_wv():
	chdir()
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
	ref = np.load('ref/plot_variance_wv.npy')
	np.testing.assert_array_equal(out, ref)


def test__plot_variance_v():
	chdir()
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
	ref = np.load('ref/plot_variance_v.npy')
	np.testing.assert_array_equal(out, ref)


def test__plot_variance_w():
	chdir()
	f1 = lambda i, j: 1 * (-i + j)
	f2 = lambda i, j: 2 * (-i + j)
	f3 = lambda i, j: 3 * (-i + j)
	out = plot_variance(
		clar_traj,
		'output/prfstats/plot_variance_w.png',
		filt_params,
		vary_param_1=('weight_func', (f1, f2, f3)),
		legend_labels_1=('weight function', ('k=1', 'k=2', 'k=3')),
		quiet=True,
		load_saved_filts='data/clar_filts_.npy',
		heatmaps=False
	)
	out = plot_variance__extract_output(out)
	ref = np.load('ref/plot_variance_w.npy')
	np.testing.assert_array_equal(out, ref)

def test__plot_pairwise_mean_dists():
	chdir()
	dists = pairwise_mean_dists(
		clar_traj,
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=True,
		load_saved_filts='data/viol_filts_vv.npy',
	)
	ref = np.load('ref/pairwise_mean_dists.npy')
	np.testing.assert_array_equal(dists, ref)

if __name__ == '__main__':
	# test__fetch_filts_v()
	test__plot_dists_to_means()
	# test__plot_ROCs_v()
	# test__plot_dists_to_means()
	# test__plot_variance_vv()
	# test__plot_variance_vw()
	pass