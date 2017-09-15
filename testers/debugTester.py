import os
os.chdir('..')

from PRFCompare.Plots import plot_variance
from PH import Filtration, make_movie, make_PD
from config import default_filtration_params as parameter_set

from matplotlib.pyplot import ioff
ioff()

test = 3


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

		('worm_length', (500, 1000, 1500)),		 # vary param 1

		('ds_rate', (20, 30, 40)),

		load_saved_filts=False,

		time_units='samples',
		crop=(5000, 2005000),    		# (start, stop) in time_units, or 'auto'
		num_windows=5,			  		# evenly spaced

		weight_func=lambda x, y: 1,  	# no weighting (constant). see test 4 for other examples

		normalize_volume=True,
		see_samples=3,				 	# interval to build filt movies and PDs. 0 means no PDs or movies.
		quiet=True
	)
