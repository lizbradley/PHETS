import os
os.chdir('..')

import numpy as np

from config import default_filtration_params as dfp

from Signals import TimeSeries
from Signals import Trajectory
from PRFStats.ROC import L2MeanPRF_ROCs


# test_TS #
def prepare_TS():
	sig = TimeSeries(
		'unit_tests/data/40-clarinet.txt',
		crop=(1000, 10000),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)
	np.save('unit_tests/ref/TS.npy', sig.windows)


# test_Traj #
def prepare_Traj():
	sig = Trajectory(
		'unit_tests/data/ellipse-test.txt',
		crop=(100, 900),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)
	np.save('unit_tests/ref/Traj.npy', sig.windows)

# test_L2MeanPRF_ROCs #
def prepare_L2MeanPRF_ROCs():
	filt_params = dfp.copy()
	ts1 = TimeSeries(
		'unit_tests/data/40-clarinet.txt',
		crop=(75000, 180000),
		num_windows=5,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)
	ts2 = TimeSeries(
		'unit_tests/data/40-viol.txt',
		crop=(35000, 140000),
		num_windows=5,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)

	traj1 = ts1.embed(tau=32, m=2)
	traj2 = ts2.embed(tau=32, m=2)
	filt_params.update({
		'max_filtration_param': -4,
		'num_divisions': 5,
		'ds_rate': 120
	})
	out = L2MeanPRF_ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		'unit_tests/output/L2MeanPRF.png',
		filt_params,
		k=(0, 5.01, .1),
		load_saved=False,
		quiet=True
	)
	np.save('unit_tests/ref/L2MeanPRF_ROC.npy', out)


if __name__ == '__main__':
	# prepare_L2MeanPRF_ROCs()
	pass
