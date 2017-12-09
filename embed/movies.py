from plots import slide_window_frame, vary_tau_frame, compare_vary_tau_frame
from signals import TimeSeries
from utilities import remove_old_frames, print_still, frames_to_movie


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

	frames_to_movie(out_fname, frame_fname, framerate, loglevel='error')

	return traj


def vary_tau(ts, out_fname, m, tau, framerate=1):
	""" movie depicting embedding of `ts` over a range of tau"""

	trajs = [ts.embed(t, m) for t in tau]

	remove_old_frames('embed/frames/')
	frame_fname = 'embed/frames/frame%03d.png'

	for i, traj in enumerate(trajs):
		print_still('frame {} of {}'.format(i + 1, len(tau)))
		vary_tau_frame(traj, frame_fname % i)
	print ''

	frames_to_movie(out_fname, frame_fname, framerate, loglevel='error')

	return trajs


def compare_vary_tau(ts1, ts2, out_fname, m, tau, framerate=1):
	trajs1 = [ts1.embed(t, m) for t in tau]
	trajs2 = [ts2.embed(t, m) for t in tau]

	remove_old_frames('embed/frames/')
	frame_fname = 'embed/frames/frame%03d.png'

	for i, trajs in enumerate(zip(trajs1, trajs2)):
		print_still('frame {} of {}'.format(i + 1, len(tau)))
		traj1, traj2 = trajs
		compare_vary_tau_frame(traj1, traj2, frame_fname % i, tau[i])
	print ''

	frames_to_movie(out_fname, frame_fname, framerate, loglevel='error')

	return trajs1, trajs2


def compare_multi():
	raise NotImplemented
