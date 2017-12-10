import numpy as np
from boilerplate import change_dir, get_test

change_dir()

from embed.movies import *

test, start_time = get_test(set_test=4)


def out_fname():
	return 'output/embed/test_{}.mp4'.format(test)


if test == 1:
	ts = TimeSeries(
		'datasets/time_series/C135B/49-C135B.txt',
		crop=(1, 5),
		num_windows=10,
		window_length=.05,
		time_units='seconds'
	)
	slide_window(ts, out_fname(), m=2, tau=.001)

if test == 2:

	ts = TimeSeries(
		'datasets/time_series/C135B/49-C135B.txt',
		crop=(50000, 55000),
		time_units='samples'
	)
	vary_tau(ts, out_fname(), m=2, tau=range(15), framerate=10)


if test == 3:

	ts1 = TimeSeries(
		'datasets/time_series/C135B/49-C135B.txt',
		crop=(50000, 55000),
		time_units='samples'
	)

	ts2 = TimeSeries(
		'datasets/time_series/C134C/49-C134C.txt',
		crop=(50000, 55000),
		time_units='samples'
	)
	compare_vary_tau(ts1, ts2, out_fname(), m=2, tau=range(15), framerate=10)

if test == 4:

	compare_multi(
		'datasets/time_series/C134C/{:02d}-C134C.txt',
		'datasets/time_series/C135B/{:02d}-C135B.txt',
		np.arange(1, 60, 10),
		out_fname(),
		crop=(1, 1.1),
		time_units='seconds',
		m=2,
		tau=.001
	)


