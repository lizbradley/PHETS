import cPickle
import numpy as np

from paths import chdir; chdir()

from utilities import clear_dir_rf
from PRFstats import fetch_filts, plot_variance, plot_ROCs, plot_dists_to_means
from common import filt_params, ellipse_traj, viol_traj, clar_traj


def test__fetch_filts_v():
	chdir()
	clear_dir_rf('output')
	out = fetch_filts(
		ellipse_traj, filt_params, load_saved=False, quiet=True,
		vary_param_1=('ds_rate', (5, 7, 9)),
		save=False
	)
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


def test__plot_variance_vv():
	chdir()
	# clear_dir_rf('output')
	out = plot_variance(
		viol_traj,
		'output/plot_variance.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=True,
		load_saved_filts=True,
		filts_fname='data/viol_filts_vv.npy',
		unit_test=True,
		see_samples=False
	)
	ref = cPickle.load(open('ref/plot_variance_vv.p'))
	assert np.array_equal(out[0][0].variance, ref[0][0].variance)


def test__ROCs_v():
	chdir()
	# clear_dir_rf('output')
	out = plot_ROCs(
		clar_traj, viol_traj,
		'output/L2ROCs.png',
		filt_params,
		vary_param=('ds_rate', np.arange(80, 150, 10)),
		k=(0, 5.01, .1),
		quiet=True,
		load_saved_filts=True,
		filts_fnames=('data/clar_filts_v.npy', 'data/viol_filts_v.npy'),
	)
	ref = np.load('ref/plot_ROCs_v.npy')
	assert np.array_equal(out, ref)


if __name__ == '__main__':
	# test__fetch_filts_v()
	test__ROCs_v()
