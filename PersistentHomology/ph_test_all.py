import sys
import numpy as np
from PersistentHomology.TestingFunctions import build_and_save_filtration, make_filtration_movie, make_persistence_diagram
from PersistentHomology.FiltrationPlotter import make_frame3D
from PersistentHomology.TestingFunctions import parameter_set



in_data_file_name = "datasets/embedded/L63_x_m2/L63_x_m2_tau35.txt"
build_filt_params = parameter_set
build_filt_params.update(
	{
		'ds_rate': 60,
		'worm_length': 5000,
		'max_filtration_param': -5,
		# 'd_cov': 20,
		'num_divisions': 10,
		'use_cliques': True,

	})

# start_pt = 0  # fraction to skip of in data file (primitive sliding window)
# build_and_save_filtration(in_data_file_name, build_filt_params, start=start_pt)  # comment out to reuse filtration
#
# make_filtration_movie(
# 	in_data_file_name,              # used to check if saved filtration is up to date, and in titlebox
# 	"output/PersistentHomology/L63_x_m2_tau7_movie.mp4",      		# output filename
# 	build_filt_params,              # passed to BuildComplex.build_filtration()
#
# 	# the following are optional plotting parameters and may be omitted
# 	# see documentation at line 76 of TestingFunctions.py.
# 	color_scheme='none',
# 	framerate=1,
# )

make_persistence_diagram(
	in_data_file_name,
	"output/PersistentHomology/L63_x_m2_tau7_persistence_new.png",
	build_filt_params
)