import numpy as np

from boilerplate import change_dir, get_test
change_dir()

from config import default_filtration_params as filt_params
from PRFstats import plot_variance
from signals import TimeSeries


def test_input():
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

	return traj, filt_params

test = 104


##### plot_variance #####

if test == 100:     # v         # PASSING
	traj, filt_params = test_input()
	plot_variance(
		traj,
		'output/PRFstats/plot_variance_v.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		quiet=True,
		load_saved_filts=False,
		see_samples={'interval': 4, 'filt_step': 3}
	)

if test == 101:     # vv        # PASSING
	traj, filt_params = test_input()
	out = plot_variance(
		traj,
		'output/PRFstats/plot_variance_vv.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		vary_param_2=('max_filtration_param', (-5, -6, -7)),
		quiet=True,
		load_saved_filts=True,
		see_samples={'interval': 4, 'filt_step': 3}
	)

if test == 102:     # w         # PASSING
	traj, filt_params = test_input()
	f1 = lambda i, j: 1 * (-i + j)
	f2 = lambda i, j: 2 * (-i + j)
	f3 = lambda i, j: 3 * (-i + j)
	out = plot_variance(
		traj,
		'output/PRFstats/plot_variance_w.png',
		filt_params,
		vary_param_1=('weight_func', (f1, f2, f3)),
		legend_labels_1=('weight function', ('k=-10', 'k=1', 'k=10')),
		quiet=False,
		load_saved_filts=True,
		see_samples={'interval': 4, 'filt_step': 3}
	)

if test == 103:     # ww
	raise NotImplemented

if test == 104:     # vw        # PASSING
	traj, filt_params = test_input()
	f1 = lambda i, j: 1 * (-i + j)
	f2 = lambda i, j: 2 * (-i + j)
	f3 = lambda i, j: 3 * (-i + j)
	out = plot_variance(
		traj,
		'output/PRFstats/plot_variance_vw.png',
		filt_params,
		vary_param_1=('ds_rate', np.arange(100, 150, 10)),
		vary_param_2=('weight_func', (f1, f2, f3)),
		legend_labels_2=('k=1', 'k=2', 'k=3'),
		quiet=True,
		load_saved_filts=True,
		see_samples={'interval': 4, 'filt_step': 3}
	)

if test == 105:     # wv
	raise NotImplemented



