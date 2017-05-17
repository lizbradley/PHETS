import sys
import numpy as np

from PersistentHomology.TestingFunctions import parameter_set
from PRFCompare.PRF import PRF_dist_plot, mean_PRF_dist_plots

# TODO: fix embed movie titlebox
# TODO: finish auto_embed
# TODO: add PRF contour plot
# TODO: prfc weighting functions
# TODO: clarinet data
# TODO: GI Complex
# TODO: make dce take tau (sec), update all calling functions ?



# test = int(sys.argv[1])
test = 5

if test == 1:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -10,
			'num_divisions': 50,
			'use_cliques': True
		}
	)

	i_ref = 17
	i_arr = np.arange(2, 40, 2)
	direc = 'datasets/embedded/test_cases'
	base_filename = 'L63_x_m2_tau'
	filename_format = 'base i'				# 'i base' or 'base i'

	out_filename = 'output/PRFCompare/distances1.png'

	PRF_dist_plot(
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
		'datasets/time_series/C134C/40-C134C.txt',  # input (left)
		'datasets/time_series/C135B/40-C135B.txt',  # input (right)
		'output/PRFCompare/40_C134C_vs_C135B.png',  # out filename
		params,
		crop_1=(1, 2),			# seconds or 'auto'
		crop_2=(1, 2),			# seconds or 'auto'
		auto_crop_length=.3,		# seconds. length of windows when crop is 'auto'
		window_size=.05, 		# seconds
		num_windows=10, 		# evenly spaced
		mean_samp_num=10,  		# number of windows to use for mean
		tau=.0012,		  		# seconds
		tau_T=np.pi,			# tau_T = tau / period
		note_index=40,			# required for auto tau
		PD_movie_int=0,  		# interval to build filt movies and PDs. 0 means no PDs or movies.


	)


if test == 3:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -10,
			'num_divisions': 30,
			'use_cliques': True,

		}
	)

	mean_PRF_dist_plots(
		'datasets/time_series/C134C/40-C134C.txt',  # input (left)
		'datasets/time_series/C135B/40-C135B.txt',  # input (right)
		'output/PRFCompare/40_C134C_vs_C135B_exp_crop_usecliques_detect.png',  # out filename
		params,

		crop_1=(1, 1.3),			# seconds or 'auto'
		crop_2=(1, 1.3),			# seconds or 'auto'
		auto_crop_length=.3,		# seconds. length of windows when crop is 'auto'

		window_size=.1, 			# seconds
		num_windows=6, 				# evenly spaced
		mean_samp_num=5,  			# number of windows to use for mean

		tau='auto ideal',			# seconds or 'auto ideal' or 'auto detect'
		tau_T=np.pi,				# tau_T = tau / period
		note_index=40,				# required for auto tau
		PD_movie_int=0,  			# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 4:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -10,
			'num_divisions': 50,
			'use_cliques': True
		}
	)

	i_ref = 15
	i_arr = np.arange(10, 20, 1)
	direc = 'datasets/embedded/test_cases'
	base_filename = 'L63_x_m2_tau'
	filename_format = 'base i'				# 'i base' or 'base i'

	out_filename = 'output/PRFCompare/PRF_dist_plots.png'

	PRF_dist_plot(
		direc, base_filename, filename_format, out_filename, i_ref, i_arr, params,
		PD_movie_int=0		# interval to build filt movies and PDs. 0 means no PDs or movies.
	)

if test == 5:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': .01,
			'num_divisions': 30,
			'use_cliques': True,

		}
	)

	mean_PRF_dist_plots(
		'datasets/time_series/C134C/40-C134C.txt',  # input (left)
		'datasets/time_series/C135B/40-C135B.txt',  # input (right)
		'output/PRFCompare/mean_PRF_dist_plots.png',  # out filename
		params,

		crop_1=(1, 1.3),			# seconds or 'auto'
		crop_2=(1, 1.3),			# seconds or 'auto'
		auto_crop_length=.3,		# seconds. length of windows when crop is 'auto'

		window_size=.1, 			# seconds
		num_windows=6, 				# evenly spaced
		mean_samp_num=5,  			# number of windows to use for mean

		tau='auto ideal',			# seconds or 'auto ideal' or 'auto detect'
		tau_T=np.pi,				# tau_T = tau / period
		note_index=40,				# required for auto tau
		PD_movie_int=0,  			# interval to build filt movies and PDs. 0 means no PDs or movies.

	)