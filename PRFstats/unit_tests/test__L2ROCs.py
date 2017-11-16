import numpy as np

from PRFstats import L2ROCs
from utilities import clear_dir_force

from paths import chdir
chdir()

from common import filt_params, viol_traj, clar_traj


def test__L2ROCs_v():
	chdir()
	clear_dir_force('output')
	out = L2ROCs(
		clar_traj, viol_traj,
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
