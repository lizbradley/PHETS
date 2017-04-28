import sys
import numpy as np

from PersistentHomology.TestingFunctions import parameter_set
from PRFCompare.PRF import PRF_dist_plots, mean_PRF_dist_plots


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
			'num_divisions' : 50
		}
	)


	mean_PRF_dist_plots(
		'datasets/embedded/test_cases/viol/36-viol.txt',
		'datasets/embedded/test_cases/C134C/36-C134C.txt',
		'output/PRFCompare/dist_test_left.png',
		params,
		mean_from='left'
	)

	mean_PRF_dist_plots(
		'datasets/embedded/test_cases/viol/36-viol.txt',
		'datasets/embedded/test_cases/C134C/36-C134C.txt',
		'output/PRFCompare/dist_test_right.png',
		params,
		mean_from='right'
	)