import numpy as np
import sys, os

from PRFstats import L2ROCs
from config import default_filtration_params as filt_params
from utilities import clear_dir_force

from paths import chdir
chdir()

from common import viol_traj, clar_traj


def test__L2ROCs_v():
	chdir()
	print os.getcwd()
	clear_dir_force('output')
	out = L2ROCs(
		clar_traj, viol_traj,
		'clarinet', 'viol',
		'output/L2ROCs.png',
		filt_params,
		vary_param=('ds_rate', np.arange(80, 150, 10)),
		k=(0, 5.01, .1),
		quiet=True,
		load_saved_filts=True,
		filts_fnames=('data/clar_filts_v.npy', 'data/viol_filts_v.npy'),
	)
	ref = np.load('ref/L2ROCs_v.npy')
	assert np.array_equal(out, ref)
