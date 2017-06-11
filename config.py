WAV_SAMPLE_RATE = 44100.

MEMORY_PROFILE_ON = True

find_landmarks_c_compile_str = {
	'linux': '/usr/bin/gcc-5 -fopenmp -lpopt find_landmarks.c -o find_landmarks -lm -lpopt',

	'macOS': '/usr/local/bin/gcc-5 -fopenmp -lpopt -o find_landmarks find_landmarks.c'
}


default_filtration_params = {
	"num_divisions": 50,
	"max_filtration_param": -20,
	"min_filtration_param": 0,
	"start": 0,
	"worm_length": 10000,
	"ds_rate": 50,
	"landmark_selector": "maxmin",
	"use_ne_for_maxmin": False,
	"d_speed_amplify": 1,
	"d_orientation_amplify": 1,
	"d_stretch": 1,
	"d_ray_distance_amplify": 1,
	"d_use_hamiltonian": 0,
	"d_cov" : 0,
	"simplex_cutoff": 0,
	"weak": False,
	"absolute": False,
	"use_cliques": False,
	"use_twr": False,
	"m2_d": 0, 					 #Set to anything but 0 to run, set 'time_order_landmarks' = TRUE (don't think i need last part anymore - CHECK)
	"straight_VB": 0,
	"out": None,
	"program": "Perseus",
	"dimension_cutoff": 2,
	"time_order_landmarks": False,
	"connect_time_1_skeleton": False,
	"reentry_filter": False,
	"store_top_simplices": True,
	"sort_output": False,
	"graph_induced": False    # Use graph induced complex to build filtration.
}
