import sys
import numpy as np
from config import default_filtration_params as parameter_set
# from PRFCompare.PRF import PRF_dist_plot, mean_PRF_dist_plots
from PRFCompare.PRFCompare import plot_dists_vs_means, plot_dists_vs_ref, plot_clusters



# test = int(sys.argv[1])
test = 1001

if test == 1:
	params = parameter_set
	# params = standard_parameter_set
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
		'datasets/embedded/test_cases', 		 # input directory
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

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 2:
	params = parameter_set
	# params = standard_parameter_set
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
		auto_crop_length=.3,	# seconds. length of windows when crop is 'auto'

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

	plot_dists_vs_means(
		'datasets/time_series/C134C/40-C134C.txt',  # input (left)
		'datasets/time_series/C135B/40-C135B.txt',  # input (right)
		'output/PRFCompare/40_C134C_vs_C135B.png',  # out filename
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
			'ds_rate': 57,
			'worm_length': 5000,
			'max_filtration_param':-5,
			'num_divisions': 20,
			'use_cliques': True
		}
	)


	plot_dists_vs_ref(			# previously PRF_dist_plot()
		'datasets/embedded/test_cases',						# input directory
		'L63_x_m2_tau',										# input base filename
		'base i',											# input filename format: 'base i or 'i base'
		'output/PRFCompare/ref/test4_L1.png',			# output filename
		params,

		load_saved_filtrations=False,

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
	# params = standard_parameter_set
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

		load_saved_filtrations=False,

		crop_1='auto',					# seconds or 'auto'
		crop_2='auto',					# seconds or 'auto'
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

		PD_movie_int=0,  				# interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 6:
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
		'datasets/embedded/L63_x_m2/L63_x_m2_tau18.txt',
		'datasets/embedded/L63_x_m2/L63_x_m2_tau35.txt',
		'output/PRFCompare/mean/L63_tau18v35_W1000dsr50.png',  		# out filename
		params,

		load_saved_filtrations=False,

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

		PD_movie_int=5,  				# interval to build filt movies and PDs. 0 means no PDs or movies.

	)
	
if test == 7:
	params = parameter_set
	# params = standard_parameter_set
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
		'datasets/embedded/L63_x_m2/L63_x_m2_tau18.txt',
		'datasets/embedded/L63_x_m2/L63_x_m2_tau30.txt',
		'output/PRFCompare/mean/L63_x_m2_tau18_v_tau30.png',  		# out filename
		params,

		load_saved_filtrations=True,
		time_units='seconds',

		crop_1=(.1, .2),				# seconds or 'auto'
		crop_2=(.1, .2),				# seconds or 'auto'
		auto_crop_length=.5,			# seconds. length of windows when crop is 'auto'

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

		PD_movie_int=0,  				# interval to build filt movies and PDs. 0 means no PDs or movies.

	)

if test == 11:
	params = parameter_set
	# params = standard_parameter_set
	params.update(
		{
			'ds_rate': 75,
			'max_filtration_param': -10,
			'num_divisions': 20,
			'use_cliques': True,

		}
	)

	plot_dists_vs_means(		# renamed from mean_PRF_dists_plot()
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L.png',  # out filename
		params,

		load_saved_filtrations=False,

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

		PD_movie_int=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L.png',  # out filename
		params,

		load_saved_filtrations=True,

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

		PD_movie_int=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

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

		load_saved_filtrations=False,

		time_units='seconds',

		crop_1=(2, 2.5),  # time_units or 'auto'
		crop_2=(2, 2.5),  # time_units or 'auto'
		auto_crop_length=1,

		window_size=.1,  # time_units
		num_windows=10,  # evenly spaced
		mean_samp_num=10,  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=20,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		PD_movie_int=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)


if test == 14:

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
		'datasets/time_series/viol/49-viol.txt',
		'datasets/time_series/C134C/49-C134C.txt',
		'output/PRFCompare/mean/viol_C134C.png',  		# out filename
		params,

		load_saved_filtrations=False,

		time_units='seconds',

		crop_1=(2, 2.5),  # time_units or 'auto'
		crop_2=(2, 2.5),  # time_units or 'auto'
		auto_crop_length=1,

		window_size=.1,  # time_units
		num_windows=10,  # evenly spaced
		mean_samp_num=10,  # number of windows to use for mean


		weight_func=lambda i, j: 1,  # no weighting (constant). see test 4 for other examples

		PRF_res=50,  # num divisions

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', 'b', or 'a + b'
		# a is magnitude of window PRF, b is magnitude of ref PRF

		normalize_volume=True,

		see_samples=1,  # interval to build filt movies and PDs. 0 means no PDs or movies.

	)

	
	
############ ACTUAL TEST EXPLORATION TIME  !!! #########


### using 1000 witnesses, 100 landmarks: looking at normalization of variance by magnitude of mean
	
if test == 50:
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_no_weight.png',  # out filename
		params,

		load_saved_filtrations=False,

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

		#PD_movie_int=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_lin.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_exp.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_1000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
	
######### 2000 witnesses, 100 landmarks:

	
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_no_weight.png',  # out filename
		params,

		load_saved_filtrations=False,

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

		#PD_movie_int=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_lin.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_exp.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_2000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
	

######## 5000 W, 100 L:
	
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_no_weight.png',  # out filename
		params,

		load_saved_filtrations=False,

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

		#PD_movie_int=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_lin.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_exp.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_5000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
	
	
######### 10,000 W and 100 L: 

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_no_weight.png',  # out filename
		params,

		load_saved_filtrations=False,

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

		#PD_movie_int=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_lin.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_exp.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_10000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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

############# 20000 W , 100 L

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_no_weight.png',  # out filename
		params,

		load_saved_filtrations=False,

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

		#PD_movie_int=5,  # interval to build filt movies and PDs. 0 means no PDs or movies.
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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_no_weight_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_lin.png',  # out filename
		params,

		load_saved_filtrations=False,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_lin_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_exp.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'datasets/embedded/REALDEAL/L63_2mil.txt',
		'output/PRFCompare/mean/L63_20000W100L_exp_scaled.png',  # out filename
		params,

		load_saved_filtrations=True,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/ref/L63_vary_tau_Euclidean_none.png', 		 # output filename
		params,
		
		load_saved_filtrations=False,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_none_scaled.png', 		 # output filename
		params,
		
		#load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1, 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_lin.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_lin_scaled.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 5 * (j - i), 						# linear, k = .1

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_exp.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_Euclidean_exp_scaled.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) , 						# linear, k = .1

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_none.png', 		 # output filename
		params,
		
		#load_saved_filtrations=False,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_none_scaled.png', 		 # output filename
		params,
		
		#load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1, 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_lin.png', 		 # output filename
		params,
		
		load_saved_filtrations=False,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_lin_scaled.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 5 * (j - i), 							# linear, k = .1

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_exp.png', 		 # output filename
		params,
		
		load_saved_filtrations=False,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_nH10_exp_scaled.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)), 							# linear, k = .1

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_none.png', 		 # output filename
		params,
		
		#load_saved_filtrations=False,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1,  						# linear, k = .1

		dist_scale='none',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_none_scaled.png', 		 # output filename
		params,
		
		#load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 1,  						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_lin.png', 		 # output filename
		params,
		
		load_saved_filtrations=False,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_lin_scaled.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func=lambda i, j: 5 * (j - i), 						# linear, k = .1

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_exp.png', 		 # output filename
		params,
		
		load_saved_filtrations=False,

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
		'datasets/embedded/test_cases', 		 # input directory
		'L63_x_m2_tau', 						 # input base filename
		'base i', 								 # input filename format: 'base i or 'i base'
		'output/PRFCompare/L63_vary_tau_O20_exp_scaled.png', 		 # output filename
		params,
		
		load_saved_filtrations=True,

		i_ref=18,
		i_arr=np.arange(2, 40, 1),

		# weight_func= lambda i, j: np.power(np.e, .1 * (j - i)) 		# exponential k = .1
		weight_func= lambda i, j: np.power(np.e, 5 * (j - i)) , 						# linear, k = .1

		dist_scale='b',							# 'none', 'a', or 'a + b'
		PRF_res=25,  								# num divisions

		see_samples=0,  							# interval to build filt movies and PDs. 0 means no PDs or movies.

	)