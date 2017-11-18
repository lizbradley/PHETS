from boilerplate import change_dir, get_test

change_dir()

from signals import TimeSeries
from DCE.movies import slide_window

test, start_time = get_test(set_test=1)

def out_fname():
	return 'output/DCE/test_{}.mp4'.format(test)

if test == 1:
	ts = TimeSeries(
		'datasets/time_series/C135B/49-C135B.txt',
		crop=(1, 5),
		num_windows=10,
		window_length=.05,
		time_units='seconds'
	)
	traj = slide_window(ts, out_fname(), m=2, tau=.001)
