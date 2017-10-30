import os, sys
import numpy as np

from config import default_filtration_params as dfp


from signals import TimeSeries
from signals import Trajectory
from PRFstats.interface import L2ROCs


def test_TS():

	sig = TimeSeries(
		'unit_tests/data/40-clarinet.txt',
		crop=(1000, 10000),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)

	assert np.array_equal(sig.windows, np.load('unit_tests/ref/TS.npy'))



def test_Traj():
	sig = Trajectory(
		'unit_tests/data/ellipse-test.txt',
		crop=(100, 900),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)
	assert np.array_equal(sig.windows, np.load('unit_tests/ref/Traj.npy'))


def test_L2MeanPRF_ROCs():
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
		'unit_tests/output/L2MeanPRF.png',
		filt_params,
		k=(0, 5.01, .1),
		load_saved_filts=False,
		quiet=True
	)
	assert np.array_equal(out, np.load('unit_tests/ref/L2MeanPRF_ROC.npy'))
