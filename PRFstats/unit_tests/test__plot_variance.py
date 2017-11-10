import cPickle
import numpy as np
import os

from PRFstats import plot_variance
from config import default_filtration_params as filt_params
from utilities import clear_dir_force

from common import current_dir, root_dir
from trajs import viol_traj



def test__plot_variance():

	os.chdir(current_dir)
	print os.getcwd()
	clear_dir_force('output')

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

	cPickle.dump(out, open('output/plot_variance.p', 'wb'))
	ref = cPickle.load(open('ref/plot_variance.p'))

	os.chdir(root_dir)

	assert np.array_equal(out[0][0].variance, ref[0][0].variance)

