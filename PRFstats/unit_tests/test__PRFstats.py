import cPickle
import numpy as np

from paths import chdir; chdir()

from utilities import clear_dir_rf
from PRFstats import filt_set, plot_variance, plot_ROCs, plot_dists_to_means
from common import filt_params, ellipse_traj, viol_traj, clar_traj, \
	plot_variance__extract_output


def test__fetch_filts_v():
	chdir()
	clear_dir_rf('output')
	out = filt_set(ellipse_traj, filt_params,
	               vp1=('ds_rate', (5, 7, 9)), load_saved=False,
	               quiet=True, save=False)
	ref = np.load('ref/fetch_filts_v.npy')

	same = True
	for index, f_ref in np.ndenumerate(ref):
		if not np.array_equal(f_ref.complexes, out[index].complexes):
			same = False
			break

	assert same

def test__plot_dists_to_means():
	chdir()
	out = plot_dists_to_means(
		clar_traj, viol_traj,
		'output/dists_to_means.png',
		filt_params,
		# weight_func=lambda x, y: x ** 2 + y ** 2,
		load_saved_filts=True,
		filts_fnames=('data/clar_filts_.npy', 'data/viol_filts_.npy')
	)

def test__plot_variance_vw():
	chdir()

	f1 = lambda i, j: .1 * (i + j)
	f2 = lambda i, j: .2 * (i + j)
	f3 = lambda i, j: .3 * (i + j)

	out = plot_variance(
		clar_traj,
		'output/plot_variance_vw.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		vary_param_2=('weight_func', (f1, f2, f3)),
		legend_labels_2=('k=.1', 'k=.2', 'k=.3'),
		quiet=True,
		load_saved_filts=True,
		filts_fname='data/clar_filts_v.npy',
		unit_test=True,
		see_samples=False
	)
	assert True

def test__plot_variance_vv():
	chdir()
	# clear_dir_rf('output')
	out = plot_variance(
		viol_traj,
		'output/plot_variance_vv.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=True,
		load_saved_filts=True,
		filts_fname='data/viol_filts_vv.npy',
		unit_test=True,
		see_samples=False
	)
	out = plot_variance__extract_output(out)
	ref = np.load('ref/plot_variance_vv.npy')
	np.testing.assert_array_equal(out, ref)


def test__ROCs_v():
	chdir()
	# clear_dir_rf('output')
	out = plot_ROCs(
		clar_traj, viol_traj,
		'output/ROCs.png',
		filt_params,
		vary_param=('ds_rate', np.arange(80, 150, 10)),
		k=(0, 5.01, .1),
		quiet=True,
		load_saved_filts=True,
		filts_fnames=('data/clar_filts_v.npy', 'data/viol_filts_v.npy'),
	)
	ref = np.load('ref/plot_ROCs_v.npy')
	np.testing.assert_array_equal(out, ref)


if __name__ == '__main__':
	# test__fetch_filts_v()
	# test__plot_dists_to_means()
	# test__ROCs_v()
	# test__plot_dists_to_means()
	# test__plot_variance_vv()
	test__plot_variance_vw()
	pass