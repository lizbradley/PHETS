import os, sys

import numpy as np

from config import default_filtration_params as dfp

from signals import TimeSeries
from signals import Trajectory
from PRFstats.interface import L2ROCs

os.chdir('..')
sys.path.append(os.path.dirname(__file__))

def prepare_TS():
	sig = TimeSeries(
		'unit_tests/data/40-clarinet.txt',
		crop=(1000, 10000),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)
	np.save('unit_tests/ref/TS.npy', sig.windows)


def prepare_Traj():
	sig = Trajectory(
		'unit_tests/data/ellipse-test.txt',
		crop=(100, 900),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)
	np.save('unit_tests/ref/Traj.npy', sig.windows)

def prepare_L2ROCs():
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
	out = L2ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		'unit_tests/output/L2ROCs.png',
		filt_params,
		k=(0, 5.01, .1),
		load_saved_filts=False,
		quiet=True
	)
	np.save('unit_tests/ref/L2ROCs.npy', out)


if __name__ == '__main__':
	prepare_L2ROCs()
	pass

