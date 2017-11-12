import cPickle
import numpy as np

import os, sys

from paths import root_dir, current_dir

sys.path.append(root_dir)
os.chdir(current_dir)

from common import filt_params, viol_traj, clar_traj, ellipse_traj

from utilities import clear_dir_force
from PRFstats import plot_variance, L2ROCs
from PRFstats.data import fetch_filts


def filts2comps(fname):
	filts = np.load(fname)

	comps = np.zeros_like(filts)
	for idx, f in np.ndenumerate(filts):
		comps[idx] = f.complexes

	out_fname = fname.replace('filts', 'comps')
	np.save(out_fname, comps)


def ref__fetch_filts_v():
	fetch_filts(
		ellipse_traj, filt_params, load_saved=False, quiet=False,
		out_fname='ref/ellipse_filts_v.npy',
		vary_param_1=('ds_rate', (5, 7, 9))
	)

def data__viol_filts_v():
	fetch_filts(
		viol_traj, filt_params, load_saved=False, quiet=False,
		out_fname='data/viol_filts_v.npy',
		vary_param_1=('ds_rate', np.arange(80, 150, 10))
	)

def data__clar_filts_v():
	fetch_filts(
		clar_traj, filt_params, load_saved=False, quiet=False,
		out_fname='data/clar_filts_v.npy',
		vary_param_1=('ds_rate', np.arange(80, 150, 10))
	)

def data__viol_filts_vv():
	fetch_filts(
		viol_traj, filt_params, load_saved=False, quiet=False,
		out_fname='data/viol_filts_vv.npy',
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
	)

def ref__L2ROCs():

	out = L2ROCs(
		clar_traj, viol_traj,
		'clarinet', 'viol',
		'output/L2ROCs.png',
		filt_params,
		vary_param=('ds_rate', np.arange(80, 150, 10)),
		k=(0, 5.01, .1),
		quiet=False,
		load_saved_filts=True,
		filts_fnames=('data/clar_filts_v.npy', 'data/viol_filts_v.npy'),
	)
	np.save('ref/L2ROCs_v.npy', out)


def ref__plot_variance():

	clear_dir_force('output')

	out = plot_variance(
		viol_traj,
		'output/plot_variance.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=False,
		unit_test=True,
		see_samples=False,
		load_saved_filts=True,
		filts_fname='data/viol_filts_vv.npy',
	)

	cPickle.dump(out, open('ref/plot_variance_vv.p', 'wb'))


if __name__ == '__main__':
	# ref__fetch_filts_v()
	print 'HDFSDFLSDFSFSFS'
	data__viol_filts_v()
	data__clar_filts_v()
	data__viol_filts_vv()
	ref__L2ROCs()
	ref__plot_variance()
