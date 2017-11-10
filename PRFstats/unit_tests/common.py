from signals import TimeSeries
from config import default_filtration_params as filt_params

filt_params.update({
	'ds_rate': 100,
	'num_divisions': 10,
	'max_filtration_param': -8
})

clar_traj = TimeSeries(
	'data/40-clarinet.txt',
	crop=(75000, 180000),
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