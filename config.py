SAMPLE_RATE = 44100.

MEMORY_PROFILE_ON = False

find_landmarks_c_compile_str = {
    # 'linux': 'gcc -g -fopenmp -lpopt -lm -o find_landmarks find_landmarks.c',
	'linux': 'gcc -g -fopenmp -lm -o find_landmarks find_landmarks.c',

	'macOS': '/usr/local/bin/gcc-5 -fopenmp -lpopt -o find_landmarks find_landmarks.c'
}

gnuplot_str = 'gnuplot'

default_filtration_params = {
	"num_divisions": 50,
	"max_filtration_param": -20,
	"min_filtration_param": 0,
	"start": 0,
	"worm_length": None,
	"ds_rate": 50,
	"landmark_selector": "maxmin",
	"use_ne_for_maxmin": False,
	"d_speed_amplify": 1,
	"d_orientation_amplify": 1,
	"d_stretch": 1,
	"d_ray_distance_amplify": 1,
	"d_use_hamiltonian": 1,
	"d_cov" : 0,
	"simplex_cutoff": 0,
	"weak": False,
	"absolute": False,
	"use_cliques": False,
	"use_twr": False,
	"m2_d": 0, 					 # Set to anything but 0 to run
	"straight_VB": 0,
	"dimension_cutoff": 2,
	"connect_time_1_skeleton": False,
	"reentry_filter": False,
	"store_top_simplices": True,
	"graph_induced": False    # Use graph induced complex to build filtration.
}
