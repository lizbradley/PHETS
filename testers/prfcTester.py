import sys, time
import numpy as np
from config import default_filtration_params as parameter_set

from PRFCompare.Plots import plot_dists_vs_ref, plot_dists_vs_means, plot_variance, plot_clusters

set_test = 8003			 # set test number here or with command line argument



if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()


if test == 1:
	params = parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':.1,
			'num_divisions': 25,
			'use_cliques': True
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/test_1.png', 		 # output filename
		params,

		i_ref=17,
		i_arr=np.arange(2, 40, 8),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: .1 * (j - i),  						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=20,  								# num divisions

		see_samples=1,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 2:
	params = parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -1,
			'num_divisions': 30
		}
	)

	plot_dists_vs_means(
		'datasets/time_series/C134C/40-C134C.txt',  # input (left)
		'datasets/time_series/C135B/40-C135B.txt',  # input (right)
		'output/PRFCompare/40_C134C_vs_C135B.png',  # out filename
		params,
		crop_1=(1, 2),			# seconds or 'auto'
		crop_2=(1, 2),			# seconds or 'auto'

		window_size=.05, 		# seconds
		num_windows=10, 		# evenly spaced
		mean_samp_num=10,  		# number of windows to use for mean

		tau=.0012,		  		# seconds

		see_samples=0,  		# interval to build filt movies and PDs. 0 means no PDs or movies.


	)

if test == 3:
	params = parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param': -10,
			'num_divisions': 30,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/time_series/C134C/40-C134C.txt',  # input (left)
		'datasets/time_series/C135B/40-C135B.txt',  # input (right)
		'output/PRFCompare/40_C134C_vs_C135B.png',  # out filename
		params,

		crop_1=(1, 1.3),			# seconds or 'auto'
		crop_2=(1, 1.3),			# seconds or 'auto'

		window_size=.1, 			# seconds
		num_windows=6, 				# evenly spaced
		mean_samp_num=5,  			# number of windows to use for mean

		tau='auto ideal',			# seconds or 'auto ideal' or 'auto detect'
		tau_T=np.pi,				# tau_T = tau / period
		note_index=40,				# required for auto tau
		see_samples=0,  			# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 4:
	params = parameter_set
	params.update(
		{
			'ds_rate': 57,
			'worm_length': 5000,
			'max_filtration_param':-5,
			'num_divisions': 20,
			'use_cliques': True
		}
	)


	plot_dists_vs_ref(			# previously PRF_dist_plot()
		'datasets/trajectories/test_cases',						# input directory
		'L63_x_m2_tau',										# input base filename
		'base i',											# input filename format: 'base i or 'i base'
		'output/PRFCompare/ref/test4_L1.png',			# output filename
		params,

		load_saved_PRFs=True,

		i_ref=15,
		i_arr=np.arange(10, 20, 2),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential, k = .1
		weight_func=lambda i, j: .1 * (j - i),  						# linear, k = .1

		metric='L1',								# 'L1' (abs) or 'L2' (euclidean)
		dist_scale='a + b', 						# 'none', 'a', or 'a + b'

		PRF_res=20,									# num divisions

		see_samples=0,								# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 5:
	params = parameter_set
	params.update(
		{
			'ds_rate': 30,
			'max_filtration_param': -5,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/time_series/C134C/49-C134C.txt',  # input (left)
		'datasets/time_series/C135B/49-C135B.txt',  # input (right)
		'output/PRFCompare/mean/test_5.png',  		# out filename
		params,

		load_saved_PRFs=False,

		time_units='seconds',

		crop_1=(1, 2),					# seconds or 'auto'
		crop_2=(1, 2),					# seconds or 'auto'
		auto_crop_length=.5,			# seconds. length of windows when crop is 'auto'

		window_size=.05, 				# seconds
		num_windows=5, 					# evenly spaced
		mean_samp_num=5,  				# number of windows to use for mean

		tau_1='auto ideal',				# seconds or 'auto ideal' or 'auto detect'
		tau_2='auto ideal',
		tau_T=np.pi,					# tau_T = tau / period
		note_index=49,					# required for auto tau

		weight_func=lambda i, j: 1,		# no weighting (constant). see test 4 for other examples

		PRF_res=50,						# num divisions

		metric='L2', 					# 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  			# 'none', 'a', 'b', or 'a + b'
										# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=1,  				# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 6:
	params = parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau18.txt',
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau35.txt',
		'output/PRFCompare/mean/L63_tau18v35_W1000dsr50.png',  		# out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(100, 8000),					# time_units or 'auto'
		crop_2=(100, 8000),					# time_units or 'auto'

		window_size= 2000, 				# time_units
		num_windows=10, 				# evenly spaced
		mean_samp_num=10,  				# number of windows to use for mean


		weight_func=lambda i, j: 1,		# no weighting (constant). see test 4 for other examples

		PRF_res=50,						# num divisions

		metric='L2', 					# 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  			# 'none', 'a', 'b', or 'a + b'
										# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=5,  				# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 7:
	params = parameter_set
	params.update({
		'worm_length': 2000,
		'ds_rate': 10,
		'max_filtration_param': -5,
		'num_divisions': 20,
		'use_cliques': True,
		})

	plot_dists_vs_means(
		# 'datasets/time_series/C134C/49-C134C.txt',  # input (left)
		# 'datasets/time_series/C135B/49-C135B.txt',  # input (right)
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau18.txt',
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau30.txt',
		'output/PRFCompare/mean/L63_x_m2_tau18_v_tau30.png',  		# out filename
		params,

		load_saved_PRFs=True,
		time_units='seconds',

		crop_1=(.1, .2),				# seconds or 'auto'
		crop_2=(.1, .2),				# seconds or 'auto'

		window_size=.05, 				# seconds
		num_windows=10, 					# evenly spaced
		mean_samp_num=5,  				# number of windows to use for mean

		tau_1=.001,						# seconds or 'auto ideal' or 'auto detect'
		tau_2=.001,

		weight_func=lambda i, j: 1,  	# no weighting (constant). see test 4 for other examples

		PRF_res=50,						# num divisions

		metric='L2', 					# 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  			# 'none', 'a', 'b', or 'a + b'
										# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  				# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 11:
	params = parameter_set
	params.update(
		{
			'ds_rate': 75,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(		# renamed from mean_PRF_dists_plot()
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=7500,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 12:
	params = parameter_set
	params.update(
		{
			'ds_rate': 75,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)
	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 50000),  # time_units or 'auto'
		crop_2=(1000, 50000),  # time_units or 'auto'

		window_size=5000,  # time_units
		num_windows=7,  # evenly spaced
		mean_samp_num=7,  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,
		normalize_win_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 13:
	params = parameter_set
	params.update(
		{
			'ds_rate': 75,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	params = parameter_set
	params.update(
		{
			'ds_rate': 75,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_clusters(
		'datasets/time_series/viol/49-viol.txt',
		'datasets/time_series/C134C/49-C134C.txt',
		'output/PRFCompare/cluster/viol_C134C.png',  		# out filename
		params,

		load_saved_PRFs=False,

		time_units='seconds',

		crop_1=(2, 2.5),  # time_units or 'auto'
		crop_2=(2, 2.5),  # time_units or 'auto'

		window_size=.1,  # time_units
		num_windows=10,  # evenly spaced
		mean_samp_num=10,  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples


		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,
		normalize_win_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 14:

	params = parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 10,
			'use_cliques': True,

		}
	)
	plot_dists_vs_means(
		'datasets/time_series/viol/49-viol.txt',
		'datasets/time_series/C134C/49-C134C.txt',
		'output/PRFCompare/mean/viol_C134C_true.png',  		# out filename
		params,

		load_saved_PRFs=False,

		time_units='seconds',

		crop_1=(2, 2.5),  	# time_units or 'auto'
		crop_2=(2, 2.5),  	# time_units or 'auto'

		num_windows=5, 		 # evenly spaced
		mean_samp_num=5,  	 # number of windows to use for mean
		window_size=.05,	 # overrides worm_length


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=10,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,
		normalize_win_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 15:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		})


	plot_variance(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/variance/L63.png',  # out filename
		params,

		('ds_rate', np.arange(100, 180, 10)),		# vary param 1
		('use_cliques', (True, False)),				# vary param 2
		# ('use_cliques', (True,)),					# null vary param 2 example

													# For now, if you do not want to use second vary param, set this like
													# the line above: a list with one element, note the trailing comma
													# For now, both vary params must be filtration params. Working on
													# getting it to work for other params like weight_func.

		load_saved_filts=False,

		time_units='samples',

		crop=(5000, 2005000),     # (start, stop) in time_units, or 'auto'

		num_windows=10,			  # evenly spaced


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='a + b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 16:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -10,
			'num_divisions': 20,
			# 'use_cliques': True,
			'ds_rate': 80,
			'worm_length': 2000,

		})


	plot_variance(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/variance/L63_d_or_amp.png',
		params,

		('d_orientation_amplify', np.arange(0, 30, 10)),			# vary param 1
		('use_cliques', (True, False)),							# vary param 2

		load_saved_filts=True,

		time_units='samples',

		crop=(5000, 25000),		     	  # (start, stop) in time_units, or 'auto'
		num_windows=10,			 		  # evenly spaced


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=1,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=False

	)

############ ACTUAL TEST EXPLORATION TIME  !!! #########


### using 1000 witnesses, 100 landmarks: looking at normalization of variance by magnitude of mean

if test == 50:
	params = parameter_set
	params.update(
		{
			'ds_rate': 10,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_no_weight.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=1000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		#see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		see_samples = 0,

	)

if test == 51:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 10,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=1000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

######## Now to explore linear weight function with k = 5 ...
if test == 60:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 10,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_lin.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=1000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=1,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 61:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 10,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=1000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

###### Now to explore exponential weight function ...

if test == 70:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 10,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_exp.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=1000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 71:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 10,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=1000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

######### 2000 witnesses, 100 landmarks #################


if test == 150:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 20,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_no_weight.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=2000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		#see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		see_samples = 1,

	)

if test == 151:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 20,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=2000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


######## Now to explore linear weight function with k = 5 ...

if test == 160:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 20,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_lin.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=2000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 161:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 20,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=2000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

###### Now to explore exponential weight function ...

if test == 170:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 20,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_exp.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=2000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 171:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 20,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=2000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


######## 5000 W, 100 L ##############

if test == 250:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_no_weight.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=5000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		#see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		see_samples = 1,

	)

if test == 251:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=5000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


######## Now to explore linear weight function with k = 5 ...

if test == 260:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_lin.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=5000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 261:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=5000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

###### Now to explore exponential weight function ...

if test == 270:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_exp.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=5000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 271:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=5000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


######### 10,000 W and 100 L #################

if test == 350:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 100,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_no_weight.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=10000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		#see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		see_samples = 1,

	)

if test == 351:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 100,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=10000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


######## Now to explore linear weight function with k = 5 ...

if test == 360:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 100,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_lin.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=10000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 361:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 100,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=10000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


###### Now to explore exponential weight function ...

if test == 370:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 100,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_exp.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=10000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 371:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 100,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=10000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

############# 20000 W , 100 L ##############

if test == 450:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 200,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_no_weight.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=20000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		#see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		see_samples = 1,

	)

if test == 451:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 200,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=20000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

######## Now to explore linear weight function with k = 5 ...

if test == 460:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 200,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_lin.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=20000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 * (j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 461:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 200,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=20000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func=lambda i, j: 5 *(j - i),

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

###### Now to explore exponential weight function ...

if test == 470:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 200,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_exp.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=20000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 471:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 200,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_PRFs=True,

		time_units='samples',

		crop_1=(1000, 2001000),  # time_units or 'auto'
		crop_2=(1000, 2001000),  # time_units or 'auto'

		window_size=20000,		  # time_units
		num_windows=20,			  # evenly spaced
		mean_samp_num=20,  		  # number of windows to use for mean


		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) ,

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

#######  BACK to L63 and varying delay tau, Euclidean vs. Normalized Hamiltonian vs. Orientation:

####### Eulidean, scaled and not scaled for no weight, linear weight, and exponential weight


if test == 1001:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,

		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/ref/L63_vary_tau_Euclidean_none.png', 		 # output filename
		params,

		load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1, 									# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=1,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1011:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,

		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_none_scaled.png', 		 # output filename
		params,

		#load_saved_PRFs=True,
		load_saved_filtrations=False,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1, 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=3,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1101:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,

		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_lin.png', 		 # output filename
		params,

		load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 5 * (j - i), 						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1111:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,

		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_SMALL_lin_scaled.png', 		 # output filename
		params,

		load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: np.divide(1, 5 * (j - i)), 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)
if test == 1201:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,

		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_exp.png', 		 # output filename
		params,

		load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)), 						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1211:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,

		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_SMALL_exp_scaled.png', 		 # output filename
		params,

		load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, -5 * (j - i)) , 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

####### L63 vary tau with Normalized Hamiltonian 10, scaled and not scaled for no weight, linear weight, and exponential weight

if test == 1002:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_use_hamiltonian': -10
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_none.png', 		 # output filename
		params,

		#load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1, 						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1012:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_use_hamiltonian': -10
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_none_scaled.png', 		 # output filename
		params,

		#load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1, 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=3,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 1102:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_use_hamiltonian': -10
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_lin.png', 		 # output filename
		params,

		load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 5 * (j - i), 							# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=1,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1112:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_use_hamiltonian': -10
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_SMALL_lin_scaled.png', 		 # output filename
		params,

		load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: np.divide(1,5 * (j - i)), 							# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1202:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_use_hamiltonian': -10
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_exp.png', 		 # output filename
		params,

		load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)), 							# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=1,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1212:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_use_hamiltonian': -10
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_SMALL_exp_scaled.png', 		 # output filename
		params,

		load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, -5 * (j - i)), 							# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

####### L63 vary tau with Orientation 20, scaled and not scaled for no weight, linear weight, and exponential weight

if test == 1003:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_orientation_amplify': 20
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_none.png', 		 # output filename
		params,

		#load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1,  						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=1,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 1013:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_orientation_amplify': 20
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_none_scaled.png', 		 # output filename
		params,

		#load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1,  						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=3,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1103:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_orientation_amplify': 20
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_lin.png', 		 # output filename
		params,

		load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 5 * (j - i),   						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=1,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 1113:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_orientation_amplify': 20
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_SMALL_lin_scaled.png', 		 # output filename
		params,

		load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: np.divide(1,5 * (j - i)), 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 1203:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_orientation_amplify': 20
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_exp.png', 		 # output filename
		params,

		load_saved_PRFs=False,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) , 	  						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=1,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 1213:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 50,
			'worm_length': 5000,
			'max_filtration_param':-10,
			'num_divisions': 25,
			'use_cliques': True,
			'd_orientation_amplify': 20
		}
	)

	plot_dists_vs_ref(
		'datasets/trajectories/test_cases', 		 # input directory
		'L63_x_m2_tau', 							 # input base filename
		'base i', 								 	# input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_SMALL_exp_scaled.png', 		 # output filename
		params,
		
		load_saved_PRFs=True,

		i_ref=18,
		i_arr=np.arange(2, 50, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, -5 * (j - i)), 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)
	
if test == 5000:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -10,
			'num_divisions': 5,
			'use_cliques': True,


		})


	plot_variance(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/variance/test.png',  # out filename
		params,

		('ds_rate', (50, 100, 200)),		# vary param 1
		('worm_length', (2500, 5000, 10000)),		# vary param 2 or None

		load_saved_filts=True,

		time_units='samples',

		crop=(5000, 2005000),     # (start, stop) in time_units, or 'auto'

		num_windows=10,			  # evenly spaced

		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=5,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.

		quiet=True,
		annot_hm=True
	)






if test == 5001:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -10,
			'num_divisions': 5,
			'use_cliques': True,
			'worm_length': 20000


		})


	plot_variance(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/variance/nHvL.png',  # out filename
		params,

		('d_use_hamiltonian', (-1, -2, -5, -10, -20, -50)),			# vary param 1
		('ds_rate', (100, 200, 400)),								# vary param 2

		load_saved_filts=False,

		time_units='samples',

		crop=(5000, 2005000),     # (start, stop) in time_units, or 'auto'

		num_windows=10,			  # evenly spaced

		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=5,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='b',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.

		quiet=True,
		annot_hm=True,
	)


if test == 5002:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -10,
			'num_divisions': 5,
			'use_cliques': True,
			'worm_length': 5000
		})

	plot_variance(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/variance/test.png',  # out filename
		params,

		('ds_rate', (50, 100, 200)),		# vary param 1

		# None,								# null vary param 2

		('weight_func',						# vary param 2 as weight function
			(
				lambda i, j: 1,
				lambda i, j: i + j,
				# lambda i, j: 3 * (i + j)
			)
		),

		legend_labels=(						# Needed only when vary param 2 is weight_func. (For legend and filenames.)
			'weight: none',
			'weight: linear k=1',
			# 'weight: linear k=3'
		),

		load_saved_filts=False,

		time_units='samples',

		crop=(5000, 2005000),     		# (start, stop) in time_units, or 'auto'

		num_windows=10,					# evenly spaced

		weight_func=lambda i, j: 1,  	# no weighting (constant). see test 4 for other examples

		PRF_res=5,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=False,
		annot_hm=True
	)

if test == 5003:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 40,
			'max_filtration_param': -8,
			'num_divisions': 30,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(
		'datasets/time_series/Clarinet/40-clarinet.txt',
		'datasets/time_series/viol/40-viol.txt',
		'output/PRFCompare/mean/test_5503.png',  # out filename
		params,

		load_saved_PRFs=False,

		time_units='seconds',

		crop_1=(2, 4),  # time_units or 'auto'
		crop_2=(1, 3),  # time_units or 'auto'

		window_size=.05,  # time_units
		num_windows=30,  # evenly spaced
		mean_samp_num=15,  # number of windows to use for mean


		weight_func=lambda i, j: 1,

		PRF_res=30,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

###Joes test


if test == 8000:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -5, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'use_cliques': True,
			'ds_rate':30

		})


	plot_variance(
		'datasets/trajectories/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/variance/joetest.png', 		 	# out filename
		params,

		('worm_length', [200, 500, 1000, 2000]),		    # vary param 1
		None,												# vary param 2

		load_saved_filts=True,

		time_units='samples',

		crop=(5000, 2005000),     # (start, stop) in time_units, or 'auto'
		num_windows=10,			  # evenly spaced


		# weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples
		weight_func=lambda x, y: y - x,



		normalize_volume=True,
		see_samples=10,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 8001:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -5, 		# no simplexes larger than 5-simplex
			'num_divisions': 20, 				# 5 complexes in the filtration
			'use_cliques': True,
			'ds_rate':30

		})


	plot_variance(
		'datasets/trajectories/btc200thou.txt',
		'output/PRFCompare/variance/testing.png', 		 	 # out filename
		params,

		('worm_length', [200, 500, 1000, 2000]),		 # vary param 1
		('weight_func',						# vary param 2 as weight function
			(
				lambda x, y: 1,
				lambda x, y: 5 * (y - x),
				lambda x, y: np.e ** (y - x)
			)
		),

		legend_labels=(						# Needed only when vary param 2 is weight_func. (For legend and filenames.)
			'weight: none',
			'weight: linear k=5',
			'weight: exponential k=1'
		),

		load_saved_filts=True,

		time_units='samples',

		crop=(5000, 2005000),     # (start, stop) in time_units, or 'auto'
		num_windows=10,			  # evenly spaced

		normalize_volume=True,

		see_samples=10,  # interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)




if test == 8002:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -5, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'ds_rate': 30

		})


	plot_variance(
		'datasets/trajectories/btc200thou.txt',
		'output/PRFCompare/variance/testing.png', 		 	 # out filename
		params,

		('worm_length', (200, 500, 1000, 2000)),		 # vary param 1
		('use_cliques', (True, False)),

		load_saved_filts=True,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)




if test == 8003:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -5, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'ds_rate': 30

		})


	plot_variance(
		'datasets/trajectories/btc200thou.txt',
		'output/PRFCompare/variance/testing.png', 		 	 # out filename
		params,

		('worm_length', (200, 500, 1000, 2000)),		 # vary param 1
		None,

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=1,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)


if test == 9000:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 4, 				# 5 complexes in the filtration
			'worm_length': 1000,

		})


	plot_variance(
		'datasets/trajectories/REALDEAL/StandardLorenz63_IC123.txt',
		'output/PRFCompare/variance/StdL63.png', 		 	 # out filename
		params,

		('ds_rate', (100, 50)),		 # vary param 1
		('d_use_hamiltonian', (1, 2, 10)),

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=5,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=5,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)

if test == 9010:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 4, 				# 5 complexes in the filtration
			'ds_rate': 100,

		})


	plot_variance(
		'datasets/trajectories/REALDEAL/StandardLorenz63_IC123.txt',
		'output/PRFCompare/variance/StdL63.png', 		 	 # out filename
		params,

		('worm_length', (1000, 2000, 5000, 10000, 20000)),		 # vary param 1
		('d_use_hamiltonian', (1, 2, 10)),

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=5,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=5,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

##### Nikki Sunday 9/24

if test == 10000:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 4, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})
		
	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_an1.txt',
		'output/PRFCompare/variance/NewHopf_an1.png', 		 	 # out filename
		params,
		('ds_rate', (10, 100, 200)),		 # vary param 1
		None,
		
		load_saved_filts=False,
		
		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10001:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 4, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_anp1.txt',
		'output/PRFCompare/variance/NewHopf_anp1.png', 		 	 # out filename
		params,

		('ds_rate', (10, 50, 100, 200)),		 # vary param 1
		None,

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10002:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_anp01.txt',
		'output/PRFCompare/variance/NewHopf_anp01.png', 		 	 # out filename
		params,

		('ds_rate', (10, 100, 200)),		 # vary param 1
		None,

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	
# Hopf - testing new WR - for max epsilon / epsilon death / norm betti 1 / GF / LF 

if test == 10010:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})
		
	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_an1.txt',
		'output/PRFCompare/variance/NewHopf_an1.png', 		 	 # out filename
		params,
		('ds_rate', (10, 100, 200)),		 # vary param 1
		('d_use_hamiltonian', (1, -1, -1.1, -2, -5, -10, -100)),
		
		load_saved_filts=False,
		
		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10011:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_anp1.txt',
		'output/PRFCompare/variance/NewHopf_anp1.png', 		 	 # out filename
		params,

		('ds_rate', (10, 100, 200)),		 # vary param 1
		('d_use_hamiltonian', (1, -1, -1.1, -2, -5, -10, -100)),

		load_saved_filts=False,

		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10012:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_anp01.txt',
		'output/PRFCompare/variance/NewHopf_anp01.png', 		 	 # out filename
		params,

		('ds_rate', (10, 100, 200)),		 # vary param 1
		('d_use_hamiltonian', (1, -1, -1.1,-2, -5, -10, -100)),

		load_saved_filts=False,

		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10014:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})
		
	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_a0.txt',
		'output/PRFCompare/variance/NewHopf_a0.png', 		 	 # out filename
		params,
		('ds_rate', (10, 100, 200)),		 # vary param 1
		('d_use_hamiltonian', (1, -1,-1.1, -2, -5, -10, -100)),
		
		load_saved_filts=False,
		
		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10015:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_app01.txt',
		'output/PRFCompare/variance/NewHopf_ap01.png', 		 	 # out filename
		params,

		('ds_rate', (10, 100, 200)),		 # vary param 1
		('d_use_hamiltonian', (1,-1, -1.1,-2, -5, -10, -100)),

		load_saved_filts=False,

		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10016:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_app1.txt',
		'output/PRFCompare/variance/NewHopf_app1.png', 		 	 # out filename
		params,

		('ds_rate', (10, 100, 200)),		 # vary param 1
		('d_use_hamiltonian', (1, -1, -1.1,-2, -5, -10, -100)),

		load_saved_filts=False,

		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
	

if test == 10017:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_ap1.txt',
		'output/PRFCompare/variance/TEST_NewHopf_ap1.png', 		 	 # out filename
		params,

		('ds_rate', (10, 100, 200)),		 # vary param 1
		('d_use_hamiltonian', (1, -1, -1.1,-2, -5, -10, -100)),

		load_saved_filts=False,

		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=10,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)


###################### trying to figure out whats going on
if test == 10018:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			#'landmark_selector': 'EST'

		})


	plot_variance(
		'datasets/trajectories/ClassicBifurcationData/NewHopf_ap1.txt',
		'output/PRFCompare/variance/NewHopf_ap1.png', 		 	 # out filename
		params,

		('ds_rate', (50, 100)),		 # vary param 1
		('d_use_hamiltonian', (1, -1, -2, -100)),

		load_saved_filts=False,

		time_units='samples',
		crop=(500, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=4,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=2,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)

print("time elapsed: %d seconds" % (time.time() - start_time))

