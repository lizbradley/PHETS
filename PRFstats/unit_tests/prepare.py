import cPickle
import numpy as np

import os, sys

from common import root_dir, current_dir

sys.path.append(root_dir)
os.chdir(current_dir)

from trajs import viol_traj

from utilities import clear_dir_force
from PRFstats import plot_variance
from PRFstats.data import fetch_filts
from config import default_filtration_params as filt_params



def prepare__plot_variance():

	clear_dir_force('output')

	filt_params.update({
		'ds_rate': 100,
		'num_divisions': 10,
		'max_filtration_param': -8
	})

	fetch_filts(
		viol_traj, filt_params, load_saved=False, quiet=True,
	    out_fname='data/viol_filts_vv.npy',
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
	    vary_param_2=('max_filtration_param', (-5, -6, -7)),
	)

	out = plot_variance(
		viol_traj,
		'output/plot_variance.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=False,
		load_saved_filts=True,
		filts_fname='data/viol_filts_vv.npy',
		unit_test=True,
		see_samples=False
	)

	cPickle.dump(out, open('ref/plot_variance.p', 'wb'))



if __name__ == '__main__':
	prepare__plot_variance()

