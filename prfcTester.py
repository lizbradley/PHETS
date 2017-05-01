import sys
import numpy as np

from PersistentHomology.TestingFunctions import parameter_set
from PRFCompare.PRF import PRF_dist_plots, mean_PRF_dist_plots
from PRFCompare.PRF import mean_PRF_dist_plots


# test = int(sys.argv[1])
test = 2

if test == 1:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate' : 50,
			'worm_length' : 5000,
			'max_filtration_param': -10,
			'num_divisions' : 50,
		})

	i_ref = 35
	i_arr = np.arange(20, 40, 2)
	dir = 'datasets/embedded/test_cases'
	base_filename = 'L63_x_m2_tau'
	out_filename = 'output/PRFCompare/distances1.png'

	PRF_dist_plots(dir, base_filename, out_filename, i_ref, i_arr, params)


if test == 2:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate' : 50,
			'worm_length' : 5000,
			'max_filtration_param': -10,
			'num_divisions' : 30
		}
	)

	mean_PRF_dist_plots(
		'datasets/time_series/34-C134C.txt',		# input (left)
		'datasets/time_series/34-C135B.txt',		# input (right)
		'output/PRFCompare/C135B_vs_C134C.png',		# out filename
		params,
		crop=(2, 2.3),			# seconds
		window_size=.05,		# seconds
		num_windows=10,			# evenly spaced
		mean_samp_num=5,		# number of windows to use for mean
		tau=50,  				# samples (44100 Hz)
		PD_movie_int=3			# interval to build filt movies and PDs

		# TODO: auto tau, auto crop, save filt movies and PDs, add PRF contour plot, normalize volume
	)


