from plots import slide_window_frame, vary_tau_frame, compare_frame
from signals import TimeSeries
from utilities import remove_old_frames, print_still, frames_to_movie
from utilities import block_print, enable_print


def slide_window(ts, out_fname, m, tau, framerate=1):
	""" movie depicting embedding of each window of `ts`.

	Parameters
	----------
	ts : TimeSeries
		pre-windowed (e.g., initalized with `num_windows` or call `slice`
		method before passing to this function).
	out_fname : str
		filename
	m : int
		embedding dimension
	tau : int or float
		embedding delay, observes `ts.time_units`

	framerate : int

	Returns
	-------
	Trajectory

	"""
	traj = ts.embed(m=m, tau=tau)
	remove_old_frames('embed/frames/')
	frame_fname = 'embed/frames/frame%03d.png'

	for i in range(traj.num_windows):
		print_still('frame {} of {}'.format(i + 1, traj.num_windows))
		slide_window_frame(traj, i, frame_fname % i)

	print ''

	frames_to_movie(out_fname, frame_fname, framerate)

	return traj


def vary_tau(ts, out_fname, m, tau, framerate=1):
	""" movie depicting embedding of `ts` over a range of tau"""

	# ts.embed() does the full signal so we'll take crop only, for efficiency
	ts_crop = TimeSeries(ts.data, name=ts.name)
	trajs = [ts_crop.embed(t, m) for t in tau]
	for t in trajs:
		t.source_ts = ts

	remove_old_frames('embed/frames/')
	frame_fname = 'embed/frames/frame%03d.png'

	for i, traj in enumerate(trajs):
		print_still('frame {} of {}'.format(i + 1, len(tau)))
		vary_tau_frame(traj, frame_fname % i)
	print ''

	frames_to_movie(out_fname, frame_fname, framerate)

	return trajs


def compare_vary_tau(ts1, ts2, out_fname, m, tau, framerate=1):
	trajs1 = [ts1.embed(t, m) for t in tau]
	trajs2 = [ts2.embed(t, m) for t in tau]

	remove_old_frames('embed/frames/')
	frame_fname = 'embed/frames/frame%03d.png'

	for i, trajs in enumerate(zip(trajs1, trajs2)):
		print_still('frame {} of {}'.format(i + 1, len(tau)))
		traj1, traj2 = trajs
		compare_frame(traj1, traj2, frame_fname % i, tau[i])
	print ''

	frames_to_movie(out_fname, frame_fname, framerate)

	return trajs1, trajs2


def compare_multi(
		path1,
		path2,
		i_arr,
		out_fname,
		crop,
		time_units,
		m,
		tau,
		framerate=1
):
	remove_old_frames('embed/frames/')
	frame_fname = 'embed/frames/frame%03d.png'

	trajs1 = []
	trajs2 = []
	for ii, i in enumerate(i_arr):

		block_print()
		ts1 = TimeSeries(path1.format(i), crop=crop, time_units=time_units)
		ts2 = TimeSeries(path2.format(i), crop=crop, time_units=time_units)
		enable_print()
		# ts.embed() does the full signal so we'll take crop only, for efficiency
		ts1_crop = TimeSeries(ts1.data, name=ts1.name, time_units=time_units)
		ts2_crop = TimeSeries(ts2.data, name=ts2.name, time_units=time_units)
		traj1 = ts1_crop.embed(tau, m)
		traj2 = ts2_crop.embed(tau, m)
		traj1.source_ts = ts1
		traj2.source_ts = ts2

		trajs1.append(traj1)
		trajs2.append(trajs2)

		print_still('frame {} of {}'.format(ii + 1, len(i_arr)))
		compare_frame(traj1, traj2, frame_fname % ii, tau)
	print ''

	frames_to_movie(out_fname, frame_fname, framerate)

	return trajs1, trajs2
