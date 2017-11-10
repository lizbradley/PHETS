from signals import TimeSeries

viol_traj = TimeSeries(
	'data/40-viol.txt',
	crop=(35000, 140000),
	num_windows=10,
	window_length=2000,
	vol_norm=(0, 0, 1)
).embed(tau=32, m=2)