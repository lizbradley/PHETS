import sys
import numpy as np

from PersistentHomology.TestingFunctions import parameter_set
from PRFCompare.PRF import PRF_dist_plots, mean_PRF_dist_plots
from PRFCompare.PRF import mean_PRF_dist_plots

# TODO: auto tau, add PRF contour plot, weighting functions


test = int(sys.argv[1])
# test = 3

if test == 1:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -10,
			'num_divisions': 50,
		}
	)

	i_ref = 35
	i_arr = np.arange(20, 40, 2)
	direc = 'datasets/embedded/test_cases'
	base_filename = 'L63_x_m2_tau'
	filename_format = 'base i'				# 'i base' or 'base i'

	out_filename = 'output/PRFCompare/distances1.png'

	PRF_dist_plots(
		direc, base_filename, filename_format, out_filename, i_ref, i_arr, params,
		PD_movie_int=0		# interval to build filt movies and PDs. 0 means no PDs or movies.
	)


if test == 2:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -10,
			'num_divisions': 30
		}
	)

	mean_PRF_dist_plots(
		'datasets/time_series/C134C/49-C134C.txt',		# input (left)
		'datasets/time_series/C135B/49-C135B.txt',		# input (right)
		'output/PRFCompare/C135B_vs_C134C.png',			# out filename
		params,
		crop_1=(2, 2.3),		# seconds or 'auto
		crop_2=(2, 2.3),		# seconds or 'auto
		window_size=.05,		# seconds
		num_windows=10,			# evenly spaced
		mean_samp_num=5,		# number of windows to use for mean
		tau=.0011,  			# seconds
		PD_movie_int=7,			# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 3:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -10,
			'num_divisions': 30
		}
	)

	mean_PRF_dist_plots(
		'datasets/time_series/C134C/49-C134C.txt',  # input (left)
		'datasets/time_series/C135B/49-C135B.txt',  # input (right)
		'output/PRFCompare/C134C_vs_C135B_new.png',  # out filename
		params,
		crop_1=(2, 2.3),		# seconds or 'auto'
		crop_2=(2, 2.3),		# seconds or 'auto'
		crop_auto_len=.3,		# seconds. length of windows when crop is 'auto'
		window_size=.05, 		# seconds
		num_windows=10, 		# evenly spaced
		mean_samp_num=10,  		# number of windows to use for mean
		tau=.001134,  			# seconds
		PD_movie_int=0,  		# interval to build filt movies and PDs. 0 means no PDs or movies.

	)
