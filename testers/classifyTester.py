import os

from DCE import embed

os.chdir('..')

import sys, time

from Signals import TimeSeries
from Classify.ROC import L2MeanPRF_ROCs
from config import default_filtration_params as filt_params
set_test = 1



if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()


def out_fname():
	return 'output/classify/test_{}.png'.format(test)


if test == 1:

	ts1 = TimeSeries(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		crop=(75000, 180000),
		num_windows=25,
		window_length=2033,
	)

	ts2 = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(35000, 140000),
		num_windows=25,
		window_length=2033,
	)

	traj1 = ts1.embed(tau=32, m=2)
	traj2 = ts2.embed(tau=32, m=2)

	filt_params.update({
		'max_filtration_param': -10,
		'num_divisions': 10
	})

	L2MeanPRF_ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		k=(0, 5.01, .01),
		filt_params=filt_params,
		tau=32,
		m=2
	)
