from boilerplate import change_dir, get_test

change_dir()

from utilities import timeit
from embed.movies import *

test, start_time = get_test(set_test=2)


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
	timeit(vary_tau)(ts, out_fname(), m=2, tau=range(50), framerate=10)


