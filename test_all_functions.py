import sys
import numpy as np
from PH.TestingFunctions import parameter_set


#
# ########################################################################################################################
# #	PH
# ########################################################################################################################
#
# from PH.TestingFunctions import build_and_save_filtration, make_filtration_movie, make_persistence_diagram
#
# in_data_file_name = "datasets/embedded/L63_x_m2/L63_x_m2_tau7.txt"
# build_filt_params = parameter_set
# build_filt_params.update(
# 	{
# 		'ds_rate': 100,
# 		'worm_length': 5000,
# 		'max_filtration_param': -10,
# 		# 'd_cov': 20,
# 		'num_divisions': 20
#
# 	})
#
# build_and_save_filtration(in_data_file_name, build_filt_params, start=0)  # comment out to reuse filtration
#
# print '\ntesting make_filtration_movie...\n'
# make_filtration_movie(
# 	in_data_file_name,  # used to check if saved filtration is up to date, and in titlebox
# 	"output/unit_tests/test1.mp4",  # output filename
# 	build_filt_params,  # passed to BuildComplex.build_filtration()
#
# 	# the following are optional plotting parameters and may be omitted
# 	# see documentation at line 76 of TestingFunctions.py.
# 	color_scheme='highlight new',
# 	framerate=1,
# )
# print '\ntesting make_persistence_diagram...\n'
# make_persistence_diagram(
# 	in_data_file_name,
# 	"output/unit_tests/test2.png",
# 	build_filt_params
# )
#
#
# ########################################################################################################################
# #	DCE
# ########################################################################################################################
#
# from DCE.DCEMovies import slide_window, compare_multi
#
# print '\ntesting slide_window()...\n'
# slide_window(
# 	'datasets/time_series/C134C/40-C134C.txt',
# 	window_size=.05,  # seconds
# 	ds_rate=1,
# 	tau=.001,  # seconds
# 	step_size=1,
# 	save_worms=False,
# )
#
# print '\ntesting compare_multi()...\n'
# compare_multi(
# 	'datasets/time_series/C134C', '-C134C.txt',
# 	'datasets/time_series/C135B', '-C135B.txt',
#
# 	i_lims=(40, 45),  # specify note range
#
# 	embed_crop_1='auto',  # seconds or 'auto'
# 	embed_crop_2='auto',  # seconds or 'auto'
# 	auto_crop_length=.08,  # seconds for when embed_crop = 'auto'
#
# 	tau='auto detect',
# 	tau_T=np.pi,  # for auto tau. tau = period * tau_T
#
# 	normalize_volume=True,
#
# 	save_worms=False,  # to output/DCE/saved_worms
# 	save_movie=True  # False for no frames, faster worm creation
#
# )

########################################################################################################################
# PRFCompare
########################################################################################################################
from PRFCompare.PRF import PRF_dist_plot, mean_PRF_dist_plots

params = parameter_set
# params = standard_parameter_set
params.update(
	{
		'ds_rate': 100,
		'worm_length': 5000,
		'max_filtration_param': -10,
		'num_divisions': 30,
		'use_cliques': True,
	}
)

i_ref = 35
i_arr = np.arange(30, 40, 2)
direc = 'datasets/embedded/test_cases'
base_filename = 'L63_x_m2_tau'
filename_format = 'base i'  # 'i base' or 'base i'
out_filename = 'output/unit_tests/PRF_dist_plots.png'

print '\ntesting PRF_dist_plots()...\n'
PRF_dist_plot(
	direc, base_filename, filename_format, out_filename, i_ref, i_arr, params,
	PD_movie_int=0  # interval to build filt movies and PDs. 0 means no PDs or movies.
)

print '\ntesting mean_PRF_dist_plots()...\n'
mean_PRF_dist_plots(
	'datasets/time_series/C134C/40-C134C.txt',  # input (left)
	'datasets/time_series/C135B/40-C135B.txt',  # input (right)
	'output/unit_tests/4mean_PRF_dist_plots.png',  # out filename
	params,
	crop_1='auto',  # seconds or 'auto'
	crop_2=(1, 1.3),  # seconds or 'auto'
	auto_crop_length=.3,  # seconds. length of windows when crop is 'auto'
	window_size=.1,  # seconds
	num_windows=10,  # evenly spaced
	mean_samp_num=10,  # number of windows to use for mean
	tau='auto detect',  # seconds or 'auto ideal' or 'auto detect'
	tau_T=np.pi,  # tau_T = tau / period
	note_index=40,  # required for auto tau
	PD_movie_int=0,  # interval to build filt movies and PDs. 0 means no PDs or movies.

)