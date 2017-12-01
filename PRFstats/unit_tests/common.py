import numpy as np

from signals import TimeSeries, Trajectory
from config import default_filtration_params as filt_params

filt_params.update({
	'ds_rate': 100,
	'num_divisions': 10,
	'max_filtration_param': -8
})

ellipse_traj = Trajectory(
	'data/ellipse.txt',
	crop=(100, 900),
	num_windows=5,
    vol_norm=(1, 1, 1)
)

clar_traj = TimeSeries(
	'data/40-clarinet.txt',
	crop=(75000, None),
    num_windows=10,
	window_length=2000,
	vol_norm=(0, 0, 1)
).embed(tau=32, m=2)

viol_traj = TimeSeries(
	'data/40-viol.txt',
	crop=(35000, 140000),
    num_windows=10,
	window_length=2000,
	vol_norm=(0, 0, 1)
).embed(tau=32, m=2)


def plot_variance__extract_output(out):
	out_ext = np.empty_like(out, dtype=object)
	for idx, ns in np.ndenumerate(out):
		out_ext[idx] = [ns.mean, ns.lvar, ns.gvar]
	return out_ext