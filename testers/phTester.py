from boilerplate import change_dir, get_test

change_dir()    # PHETS imports must come after this line

import time
import numpy as np

from embed import embed
from signals import Trajectory, TimeSeries
from phomology import Filtration, load_filtration
from config import default_filtration_params as filt_params

test, start_time = get_test(set_test=505)


def out_fname(str):
	return 'output/phomology/test_{}{}'.format(test, str)
	


if test == 14:
	in_filename = 'datasets/trajectories/49/C134C.txt'

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -10,
			'num_divisions': 10,
			'graph_induced': False
		})

	filtration = Filtration(in_filename, filt_params)

	make_movie(
		filtration,
		'output/phomology/test14.mp4',
	)

	PRF(filtration, 'output/phomology/test14_prf.png')

	PD(
		filtration,
		'output/phomology/test14_PRF.png'
	)
if test == 15:
	in_filename = 'datasets/trajectories/49/C134C.txt'

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -10,
			'num_divisions': 10,
			'graph_induced': True
		})

	filtration = Filtration(in_filename, filt_params)
	
	make_movie(
		filtration,
		'output/phomology/test15.mp4',
	)

	PRF(filtration, 'output/phomology/test15_prf.png')

	PD(
		filtration,
		'output/phomology/test15_PRF.png'
	)


if test == 16:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -10,
			'num_divisions': 10,
		})


	filtration = Filtration(in_filename, filt_params, save='test16.p')
	filtration = load_filtration('test16.p')

	make_movie(
		filtration,
		'output/phomology/test16.mp4',
	)

	PRF(filtration, 'output/phomology/test16_prf.png')

	PD(
		filtration,
		'output/phomology/test16_PRF.png'
	)





if test == 17:
	in_filename = 'datasets/trajectories/L96N22F5_x1_m2tau10.txt'


	filt_params.update(
		{
			'ds_rate': 50,
			'worm_length': 2000,
			'max_filtration_param': -10,
			'num_divisions': 10,
			# 'use_cliques': True,

		})

	filtration = Filtration(in_filename, filt_params)

	PRF(filtration, 'output/phomology/test17_prf.png')

	PD(
		filtration,
		'output/phomology/test17_PRF.png'
	)


	make_movie(
		filtration,
		'output/phomology/test17.mp4',
		alpha=.5
	)



# test = 100
if test == 100:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	traj = Trajectory(in_filename)

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': .01,
			'num_divisions': 10,
			'always_euclidean': True
		})


	filtration = Filtration(traj, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration


	filtration.movie('output/phomology/test_euc.mp4')

# test = 101
if test == 101:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': .01,
			'num_divisions': 10,
			'd_use_hamiltonion': 5,
			'always_euclidean': False,
		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/phomology/test_ham_p_5.mp4',
	)

# test = 102
# NOT IMPLEMENTED #
# if test == 102:
# 	in_filename = "datasets/trajectories/btc2milIC123.txt"
#
# 	filt_params.update(
# 		{
# 			'ds_rate': 40,
# 			'worm_length': 2000,
# 			'max_filtration_param': .01,
# 			'num_divisions': 10,
#
# 			'd_speed_amplify': 5,
#
# 		})
#
#
# 	filtration = Filtration(in_filename, filt_params)
# 	# filtration = load_saved_filtration()		# reuses previous filtration
#
# 	make_movie(
# 		filtration,
# 		'output/phomology/test_speed5.mp4',
# 	)


# test = 103
if test == 103:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': .01,
			'num_divisions': 10,

			'd_orientation_amplify': 5,

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/phomology/test_orient5.mp4',
	)


# test = 104
# NOT IMPLEMENTED #
# if test == 104:
# 	in_filename = "datasets/trajectories/btc2milIC123.txt"
#
# 	filt_params.update(
# 		{
# 			'ds_rate': 40,
# 			'worm_length': 2000,
# 			'max_filtration_param': .01,
# 			'num_divisions': 10,
#
# 			'd_stretch_amplify': 5,
#
# 		})
#
#
# 	filtration = Filtration(in_filename, filt_params)
# 	# filtration = load_saved_filtration()		# reuses previous filtration
#
# 	make_movie(
# 		filtration,
# 		'output/phomology/test_stretch5.mp4',
# 	)


# test = 105
# NOT IMPLEMENTED #
# if test == 105:
# 	in_filename = "datasets/trajectories/btc2milIC123.txt"
#
# 	filt_params.update(
# 		{
# 			'ds_rate': 40,
# 			'worm_length': 2000,
# 			'max_filtration_param': .01,
# 			'num_divisions': 10,
#
# 			'd_ray_distance_amplify': 5,
#
# 		})
#
#
# 	filtration = Filtration(in_filename, filt_params)
# 	# filtration = load_saved_filtration()		# reuses previous filtration
#
# 	make_movie(
# 		filtration,
# 		'output/phomology/test_ray5.mp4',
# 	)


# test = 106
# NOT IMPLEMENTED #
# if test == 106:
# 	in_filename = "datasets/trajectories/btc2milIC123.txt"
#
# 	filt_params.update(
# 		{
# 			'ds_rate': 40,
# 			'worm_length': 2000,
# 			'max_filtration_param': .01,
# 			'num_divisions': 10,
#
# 			'use_ne_for_maxmin': True,
#
# 		})
#
#
# 	filtration = Filtration(in_filename, filt_params)
# 	# filtration = load_saved_filtration()		# reuses previous filtration
#
# 	make_movie(
# 		filtration,
# 		'output/phomology/test_ne.mp4',
# 	)

if test == 107:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': .01,
			'num_divisions': 10,

			'd_use_hamiltonion': 0,
			'm2_d': 10,

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/phomology/test_m2d10.mp4',
	)




if test == 108:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -8,
			'num_divisions': 10,

			"d_cov": 20,

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/phomology/test_dcov+20.mp4',
	)

if test == 109:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -8,
			'num_divisions': 10,

			"d_cov": -20,

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/phomology/test_dcov-20.mp4',
	)

if test == 110:
	in_filename = "datasets/trajectories/btc2milIC123.txt"

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -10,
			'num_divisions': 10,
			'd_use_hamiltonion': 0,
			# 'use_euclid'
			'graph_induced': True
		})

	# filtration = Filtration(in_filename, filt_params)
	traj = Trajectory(in_filename)
	filtration = Filtration(traj, filt_params)
	filtration.movie('output/phomology/test_110.mp4')

	# make_movie(filtration, 'output/phomology/test_110.mp4')


if test == 200:
	in_filename = "datasets/trajectories/Ellipse200.txt"

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 200,
			'max_filtration_param': -8,
			'num_divisions': 10,
			'd_use_hamiltonian': -10

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/dham1.mp4',
	)

if test == 201:
	print 'dham -1'
	in_filename = "datasets/trajectories/Ellipse200.txt"

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 200,
			'max_filtration_param': -8,
			'num_divisions': 10,
			'd_use_hamiltonian': -1

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/dham-1.mp4',
	)

	print 'dham 1'
	in_filename = "datasets/trajectories/Ellipse200.txt"

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 200,
			'max_filtration_param': -8,
			'num_divisions': 10,
			'd_use_hamiltonian': 1

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/dham1.mp4',
	)



	print 'no dham'
	in_filename = "datasets/trajectories/Ellipse200.txt"

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 200,
			'max_filtration_param': -8,
			'num_divisions': 10,

		})


	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()		# reuses previous filtration

	make_movie(
		filtration,
		'output/nodham.mp4',
	)

if test == 202:
	# traj = Trajectory("datasets/trajectories/btc2milIC123.txt")

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -10,
			'num_divisions': 10,
			'd_use_hamiltonion': 0,
			'graph_induced': True
		})

	# filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('mp4'))
	filt.plot_pd(out_fname('_PRF.png'))
	filt.plot_prf('_prf.png')


if test == 203:
	traj = Trajectory("datasets/trajectories/L63_x_m2/L63_x_m2_tau36.txt")

	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'max_filtration_param': -10,
			'num_divisions': 10,
			'd_use_hamiltonion': 0,
			'graph_induced': True
		})

	filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('.mp4'))
	filt.plot_pd(out_fname('_PRF.png'))
	filt.plot_prf(out_fname('_prf.png'))
	
	


if test == 500:
	ts = TimeSeries(
		"datasets/time_series/C135B/49-C135B.txt",
		crop=(75900, 78000),
		num_windows=10,
		window_length=2000,
		vol_norm=(1, 1, 1)
	)

	traj = ts.embed(tau=53, m=2)

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 2200,
			'max_filtration_param': -15,
			'num_divisions': 20,
		})

	filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('.mp4'))
	filt.plot_pd(out_fname('_PD.png'))
	filt.plot_prf(out_fname('_PRF.png'))
	
## 1/7: Nikki just testing around with new witness complexes on neuron data - not great results :/ ; something so sensitive to  local changes of direct vs midrange direction ?	
if test == 501:
	ts = TimeSeries(
		'datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 12000),
		num_windows=2,
		window_length=100000,
		vol_norm=(1, 1, 1)
	)

	traj = Trajectory('datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 11000),
		num_windows=2,
		window_length=1000,
		vol_norm=(1, 1, 1)
	)

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 1000,
			'max_filtration_param': -20,
			'num_divisions': 10,
		})

	filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('.mp4'))
	filt.plot_pd(out_fname('_PD.png'))
	filt.plot_prf(out_fname('_PRF.png'))
	
if test == 502:
	ts = TimeSeries(
		'datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 12000),
		num_windows=2,
		window_length=100000,
		vol_norm=(1, 1, 1)
	)

	traj = Trajectory('datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 11000),
		num_windows=2,
		window_length=1000,
		vol_norm=(1, 1, 1)
	)

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 1000,
			'max_filtration_param': -20,
			'num_divisions': 10,
			'd_use_hamiltonian': -1.01
		})

	filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('.mp4'))
	filt.plot_pd(out_fname('_PD.png'))
	filt.plot_prf(out_fname('_PRF.png'))
	
if test == 503:
	ts = TimeSeries(
		'datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 12000),
		num_windows=2,
		window_length=100000,
		vol_norm=(1, 1, 1)
	)

	traj = Trajectory('datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 11000),
		num_windows=2,
		window_length=1000,
		vol_norm=(1, 1, 1)
	)

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 1000,
			'max_filtration_param': -20,
			'num_divisions': 10,
			'd_use_hamiltonian': -1.001
		})

	filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('.mp4'))
	filt.plot_pd(out_fname('_PD.png'))
	filt.plot_prf(out_fname('_PRF.png'))
	
	
if test == 504:
	ts = TimeSeries(
		'datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 12000),
		num_windows=2,
		window_length=100000,
		vol_norm=(1, 1, 1)
	)

	traj = Trajectory('datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 11000),
		num_windows=2,
		window_length=1000,
		vol_norm=(1, 1, 1)
	)

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 1000,
			'max_filtration_param': -20,
			'num_divisions': 10,
			'd_use_hamiltonian': -1.0001
		})

	filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('.mp4'))
	filt.plot_pd(out_fname('_PD.png'))
	filt.plot_prf(out_fname('_PRF.png'))
	
	
if test == 505:
	ts = TimeSeries(
		'datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 12000),
		num_windows=2,
		window_length=100000,
		vol_norm=(1, 1, 1)
	)

	traj = Trajectory('datasets/PHETS_NeuronVoltageDATA/I4G6_Exc_Avg_3thru7_dcem2t200.txt',
		crop=(10000, 11000),
		num_windows=2,
		window_length=1000,
		vol_norm=(1, 1, 1)
	)

	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 1000,
			'max_filtration_param': -20,
			'num_divisions': 10,
			'd_orientation_amplify': 2
		})

	filt = Filtration(traj, filt_params, save=True)
	filt = load_filtration()

	filt.movie(out_fname('.mp4'))
	filt.plot_pd(out_fname('_PD.png'))
	filt.plot_prf(out_fname('_PRF.png'))



print("time elapsed: %d seconds" % (time.time() - start_time))





