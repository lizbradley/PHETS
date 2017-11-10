
import numpy as np

from PRFstats import fetch_filts
from utilities import clear_dir_force

from paths import chdir
chdir()

from common import filt_params, ellipse_traj


def test__fetch_filts_v():
	chdir()
	clear_dir_force('output')

	out = fetch_filts(
		ellipse_traj, filt_params, load_saved=False, quiet=True,
		vary_param_1=('ds_rate', (5, 7, 9)),
		no_save=True
	)
	ref = np.load('ref/ellipse_filts_v.npy')

	same = True
	for index, f_ref in np.ndenumerate(ref):
		if not np.array_equal(f_ref.complexes, out[index].complexes):
			same = False
			break

	assert same

