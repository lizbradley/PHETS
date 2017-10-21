import os
os.chdir('..')

import sys, time

from Signals import TimeSeries, Trajectory
from PRFStats.ROC import L2MeanPRF_ROCs
from config import default_filtration_params as filt_params


set_test = 3



if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()


def out_fname():
	return 'output/PRFStats/test_{}.png'.format(test)


if test == 1:
	# replicate IDA figure 5 #

	ts1 = TimeSeries(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		crop=(75000, 180000),
		num_windows=10,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)


	ts2 = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(35000, 140000),
		num_windows=10,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)

	traj1 = ts1.embed(tau=32, m=2)
	traj2 = ts2.embed(tau=32, m=2)

	filt_params.update({
		'max_filtration_param': -21,
		'num_divisions': 20,
		'ds_rate': 20
	})

	L2MeanPRF_ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		filt_params,
		k=(0, 5.01, .01),
		load_saved=True,
		quiet=False,
		see_samples=5
	)


if test == 2:
	# testing the vary_param capabilities #

	ts1 = TimeSeries(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		crop=(75000, 180000),
		num_windows=10,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)


	ts2 = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(35000, 140000),
		num_windows=10,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)

	traj1 = ts1.embed(tau=32, m=2)
	traj2 = ts2.embed(tau=32, m=2)

	filt_params.update({
		'ds_rate': 200,
		'num_divisions': 10,
	})

	L2MeanPRF_ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		filt_params,
		k=(0, 5.01, .01),
		load_saved=True,
		quiet=True,
		vary_param=('max_filtration_param', (-3, -6)),
		see_samples=5

	)



if test == 3:
	# testing the vary_param capabilities #

	traj1 = Trajectory(
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau35.txt',
		crop=(100, 9100),
		num_windows=5,
		window_length=1500,
		vol_norm=(0, 0, 1)      # (full, crop, window)
	)


	traj2 = Trajectory(
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau50.txt',
		crop=(100, 9100),
		num_windows=5,
		window_length=1500,
		vol_norm=(0, 0, 1)
	)


	filt_params.update({
		'ds_rate': 200,
		'num_divisions': 10,
	})

	L2MeanPRF_ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		filt_params,
		k=(0, 5.01, .01),
		load_saved=False,
		quiet=True,
		vary_param=('max_filtration_param', (-3, -6)),
		see_samples=5

	)
