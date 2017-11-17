import numpy as np

from PRFstats import plot_ROCs
from utilities import clear_dir_force

from paths import chdir
chdir()

from common import filt_params, viol_traj, clar_traj


def test__ROCs_v():
	chdir()
	clear_dir_force('output')
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
	ref = np.load('ref/ROCs_v.npy')
	assert np.array_equal(out, ref)

if __name__ == '__main__':
	test__ROCs_v()