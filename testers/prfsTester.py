from boilerplate import change_dir, get_test
change_dir()

import numpy as np
from signals import TimeSeries, Trajectory
from PRFstats.interface import L2ROCs, plot_dists_to_means, plot_clusters, \
	plot_dists_to_ref, plot_variance
from config import default_filtration_params as filt_params
from utilities import idx_to_freq

test, start_time = get_test(set_test=11)


def out_fname():
	return 'output/PRFstats/test_{}.png'.format(test)


if test == 1:
	# reproduce IDA figure 5 #
	# toggle lines 86/87 in signals/data.py to reproduce #

	ts1 = TimeSeries(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		crop=(75000, 180000),
		num_windows=50,
		window_length=2000,
		vol_norm=(0, 0, 1)  # (full, crop, windows)
	)


	ts2 = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(35000, 140000),
		num_windows=50,
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

	L2ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		filt_params,
		k=(0, 5.01, .01),
		load_saved_filts=False,
		quiet=False,
		see_samples=2
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

	L2ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		filt_params,
		k=(0, 5.01, .01),
		load_saved_filts=False,
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

	L2ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		filt_params,
		k=(0, 5.01, .01),
		load_saved_filts=False,
		quiet=True,
		vary_param=('max_filtration_param', (-3, -6)),
		see_samples=5

	)


if test == 4:

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
		'max_filtration_param': -5
	})

	plot_dists_to_means(
		traj1, traj2,
		out_fname(),
		filt_params,
		quiet=False
	)

if test == 5:

	ts1 = TimeSeries(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		crop=(1, 2),
		num_windows=5,
		window_length=.01,
		vol_norm=(0, 1, 1),      # (full, crop, window)
		time_units='seconds'
	)

	ts2 = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(1, 2),
		num_windows=5,
		window_length=.01,
		vol_norm=(0, 1, 1),
		time_units='seconds'
	)

	tau = (1 / idx_to_freq(40)) / np.e
	traj1 = ts1.embed(tau=tau, m=2)
	traj2 = ts2.embed(tau=tau, m=2)

	filt_params.update({
		'ds_rate': 20,
		'num_divisions': 10,
		'max_filtration_param': -5
	})

	plot_dists_to_means(
		traj1, traj2,
		out_fname(),
		filt_params,
		quiet=False
	)


if test == 6:

	ts1 = TimeSeries(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		crop=(1, 2),
		num_windows=5,
		window_length=.01,
		vol_norm=(0, 1, 1),      # (full, crop, window)
		time_units='seconds'
	)

	ts2 = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(1, 2),
		num_windows=5,
		window_length=.01,
		vol_norm=(0, 1, 1),
		time_units='seconds'
	)

	tau = (1 / idx_to_freq(40)) / np.e
	traj1 = ts1.embed(tau=tau, m=2)
	traj2 = ts2.embed(tau=tau, m=2)

	filt_params.update({
		'ds_rate': 20,
		'num_divisions': 10,
		'max_filtration_param': -5
	})

	plot_clusters(
		traj1, traj2,
		out_fname(),
		filt_params,
		quiet=False
	)


if test == 7:

	filt_params.update({
		'max_filtration_param': -10,
		'num_divisions': 10,
		'ds_rate': 500
	})

	plot_dists_to_ref(
		'datasets/trajectories/L63_x_m2',
		'L63_x_m2_tau',
		'base i',
		out_fname(),
		filt_params,
		i_ref=15,
		i_arr=np.arange(2, 30),
		quiet=False,
		load_saved_filts=False
	)

if test == 9:
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
		'ds_rate': 100,
		'num_divisions': 10,
		'max_filtration_param': -8
	})

	L2ROCs(
		traj1, traj2,
		'clarinet', 'viol',
		out_fname(),
		filt_params,
		k=(0, 5.01, .01),
		load_saved_filts=False,
		quiet=False,
		vary_param=('d_use_hamiltonion', (-1, 1)),
		# vary_param=('d_use_hamiltonion', (1, -1)),
		see_samples=5

	)

if test == 10:
	ts = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(35000, 140000),
		num_windows=10,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)

	traj = ts.embed(tau=32, m=2)
	filt_params.update({
		'ds_rate': 100,
		'num_divisions': 10,
		'max_filtration_param': -8
	})

	plot_variance(
		traj,
		out_fname(),
		filt_params,
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		quiet=False,
		annot_hm=False,
		load_saved_filts=True
	)


if test == 11:
	ts = TimeSeries(
		'datasets/time_series/viol/40-viol.txt',
		crop=(35000, 140000),
		num_windows=10,
		window_length=2000,
		vol_norm=(0, 0, 1)
	)

	traj = ts.embed(tau=32, m=2)
	filt_params.update({
		'ds_rate': 100,
		'num_divisions': 10,
		'max_filtration_param': -8
	})

	plot_variance(
		traj,
		out_fname(),
		filt_params,
		vary_param_1=('ds_rate', np.arange(80, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=False,
		annot_hm=False,
		load_saved_filts=True
	)
