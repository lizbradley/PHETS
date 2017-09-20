import os
os.chdir('..')

from PRFCompare.Plots import plot_variance
from PH import Filtration, make_movie, make_PD
from config import default_filtration_params as parameter_set

from matplotlib.pyplot import ioff
ioff()

test = 6



def movie_fname(test, str=''):
	return 'output/debug/test_{}_{}_movie.mp4'.format(test, str)


def pd_fname(test, str=''):
	return 'output/debug/test_{}_{}_PD.png'.format(test, str)


if test == 1:
	in_fname = 'datasets/trajectories/hopf/NewHopf_app1.txt'

	params = parameter_set
	params.update(
		{
			'worm_length': 2000,
			'max_filtration_param': 1,
			'num_divisions': 10,
			'ds_rate': 50

		})

	filt = Filtration(in_fname, params)		# BUG: filtration only has 7 steps !!
	make_movie(filt, movie_fname(test), color_scheme='highlight new')
	make_PD(filt, pd_fname(test))


if test == 2:

	in_fname = 'datasets/trajectories/hopf/NewHopf_app01.txt'

	params = parameter_set
	params.update(
		{
			'worm_length': 2000,
			'max_filtration_param': 1,
			'num_divisions': 10,
			'ds_rate': 50

		})

	filt = Filtration(in_fname, params)		# BUG: filtration only has 5 steps !!
	make_movie(filt, movie_fname(test))
	make_PD(filt, pd_fname(test))



if test == 3:

	params = parameter_set
	params.update(
		{
			'max_filtration_param': -3,
			'num_divisions': 10,
			'ds_rate': 30

		})



	plot_variance(
		'datasets/trajectories/btc200thou.txt',
		'output/PRFCompare/variance/testing.png', 		 	 # out filename
		params,

		('worm_length', (500, 700, 1000)),		 # vary param 1

		('ds_rate', (20, 30, 40)),

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=5,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=3,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=False
	)


if test == 4:

	# trying to reproduce cycle crossing weirdness -- which NewHopf to use??

	params = parameter_set
	params.update(
		{
			'max_filtration_param': 1,
			'num_divisions': 10,
			'worm_length': 2000

		})


	plot_variance(
		'datasets/trajectories/hopf/NewHopf_app1.txt',
		'output/PRFCompare/variance/testing.png',
		params,

		('d_use_hamiltonion', (-1, -5, -10, -50, -100)),
		('ds_rate', (15, 20, 25)),

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),
		num_windows=5,

		weight_func=lambda x, y: 1,

		normalize_volume=True,
		see_samples=1,
		quiet=True
	)

if test == 5:
	params = parameter_set
	params.update(
		{
			'max_filtration_param': -8, 		# no simplexes larger than 5-simplex
			'num_divisions': 10, 				# 5 complexes in the filtration
			'worm_length': 2000,
			'landmark_selector': 'EST'

		})


	plot_variance(
		'ClassicBifurcations/NewHopf_anp01.txt',
		'output/PRFCompare/variance/NewHopf_anp01.png', 			# out filename
		params,

		('d_use_hamiltonian', (0, .1, 1, 10, 100)),		# vary param 1
		('ds_rate', (10, 100, 200)),

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=10,			 		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=False,
		see_samples=5,					# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)

if test == 6:
	# perseus1.txt error

	params = parameter_set
	params.update(
		{
			'num_divisions': 10,
			'worm_length': 2000,
			'ds_rate': 10
		}
	)

	plot_variance(
		'datasets/Lorenz/StandardLorenz63_IC123.txt',
		'outputs/PRFCompare/variance/Lorenztest.png',
		params,
		('max_filtration_param', (-2, -5, -10)),
		('landmark_selector', ("EST", "maxmin")),

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),  # (start, stop) in time_units, or 'auto'
		num_windows=10,  # evenly spaced

		weight_func=lambda x, y: 1,
		# no weighting (constant). see test 4 for other examples

		normalize_volume=False,
		see_samples=10,
		# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)