import sys, time
import matplotlib.pyplot as plt
import numpy as np
from config import default_filtration_params as parameter_set
from DCE.DCE import embed
from PH import Filtration, make_movie, load_saved_filtration
from PubPlots import plot_filtration_pub, plot_PD_pub, plot_waveform_sec, plot_dce_pub

set_test = 7		# set test number here or with command line argument


if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()


# from DCE.DCE import embed
# sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt')
# traj = embed(sig, tau=.01192, m=2, time_units='seconds', embed_crop=(1.72132, 1.77132), normalize=True)
# np.savetxt('datasets/IDA_PAPER/49-C135B_embedded.txt', traj)

paper_path = '/home/elliott/phets_notes/IDA 2017/figs/'
ticks = [-.05, 0, .05]

if test == 1:
	# figure 1a

	plot_waveform_sec(
		'datasets/IDA_PAPER/49-C135B.txt',
		paper_path + 'Figure1/49_C135B_time_series.png',
		crop=(1.72, 1.77)
	)



if test == 2:
	# figure 1b

	plot_dce_pub(
		'datasets/IDA_PAPER/49-C135B_embedded.txt',
		paper_path + 'Figure1/49_C135B_embedded_taudetect_525.png'
	)


if test == 3:
	# figure 2a
	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 1,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False
		})

	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()

	#
	# make_movie(
	# 	filtration,
	# 	"output/IDA_PAPER/49-C135B.mp4",
	# 	color_scheme='none',
	# )

	plot_filtration_pub(
		filtration, 2,
		paper_path + 'Figure2/epsilon_005/49_C135B_2000Wcech_ep005.png',
		landmark_size=10,
		show_eps=False,
		label='(a)'
	)
#
	# plot_PD_pub(filtration, 'output/IDA_PAPER/fig3_PD.png')




if test == 4:
	# figure 2b

	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False
		})

	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()



	plot_filtration_pub(
		filtration, 1,
		paper_path + 'Figure2/epsilon_005/49_C135B_2000W200L_ep005.png',
		landmark_size=10,
		show_eps=False,
		label='(b)'
	)


if test == 5:
	# figure 2c


	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False
		})

	filtration = Filtration('datasets/IDA_PAPER/49-C135B_embedded.txt', filt_params)
	# filtration = load_saved_filtration()



	plot_filtration_pub(
		filtration, 2,
		paper_path + 'Figure2/epsilon_005/49_C135B_2000W50L_ep005.png',
		landmark_size=10,
		show_eps=False,
		label='(c)')


if test == 50:
	fig = plt.figure(figsize=(10, 3.5), tight_layout=True, dpi=700)
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)


	# 2a

	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 1,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False
		})

	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()

	#
	# make_movie(
	# 	filtration,
	# 	"output/IDA_PAPER/49-C135B.mp4",
	# 	color_scheme='none',
	# )

	plot_filtration_pub(
		filtration, 2,
		ax1,
		landmark_size=.1,
		show_eps=False,
		label='(a)',
		ticks=ticks,
	)

	# 2b
	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False
		})

	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()



	plot_filtration_pub(
		filtration, 1,
		ax2,
		landmark_size=1,
		show_eps=False,
		label='(b)',
		ticks=ticks,

	)

	# 2c
	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False
		})

	filtration = Filtration('datasets/IDA_PAPER/49-C135B_embedded.txt', filt_params)
	# filtration = load_saved_filtration()



	plot_filtration_pub(
		filtration, 2,
		ax3,
		landmark_size=10,
		show_eps=False,
		label='(c)',
		ticks=ticks,
	)

	# ax1.set_aspect('equal')

	plt.savefig(paper_path + 'Figure2/filts.png')




if test == 6:
	# figures 3a, 3b, 3c, 3d
	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 2000,
			'min_filtration_param': .000,
			'max_filtration_param': .010,
			'num_divisions': 20,
			'use_cliques': False
		})

	# filtration = Filtration(in_filename, filt_params)
	filtration = load_saved_filtration()


	# make_movie(
	# 	filtration,
	# 	"output/IDA_PAPER/49-C135B.mp4",
	# 	color_scheme='none',
	# )

	fig = plt.figure(figsize=(8.5, 7.5), tight_layout=True, dpi=700)
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)

	plot_filtration_pub(filtration, 1, ax1, label='(a) $\epsilon = 0.001$', show_eps=False, ticks=ticks)
	plot_filtration_pub(filtration, 3, ax2, label='(b) $\epsilon = 0.002$', show_eps=False, ticks=ticks)
	plot_filtration_pub(filtration, 5, ax3, label='(c) $\epsilon = 0.003$', show_eps=False, ticks=ticks)
	plot_PD_pub(filtration, ax4, label='(d)')

	plt.savefig(paper_path + 'Figure3/figs.png')


if test == 7:

	# figures 4a, 4b, 4c, 4d

	fig = plt.figure(figsize=(8.5, 7.5), tight_layout=True, dpi=700)
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)

	# CLARINET #

	sig = np.loadtxt('datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt')
	traj = embed(sig, tau=32, m=2, time_units='samples', crop=(100000, 102205), normalize_crop=True)

	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 20,
			'worm_length': 2000,
			'min_filtration_param': 0,
			'max_filtration_param': -15,
			'num_divisions': 10,

		})

	filtration = Filtration(traj, filt_params)


	plot_filtration_pub(filtration, 8, ax1)				# 4a
	plot_PD_pub(filtration, ax3)						# 4c


	# VIOL #

	sig = np.loadtxt('datasets/time_series/viol/40-viol.txt')
	traj = embed(sig, tau=32, m=2, time_units='samples', crop=(50000, 52205), normalize_crop=True)

	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 20,
			'worm_length': 2000,
			'min_filtration_param': 0,
			'max_filtration_param': -15,
			'num_divisions': 10,

		})

	filtration = Filtration(traj, filt_params)
	plot_filtration_pub(filtration, 4, ax2)				# 4b
	plot_PD_pub(filtration, ax4)						# 4d

	plt.savefig('output/IDA_PAPER/fig4.png')


if test == 8:
	# figure 5a
	# leave as is
	pass

if test == 9:
	# figure 5b
	# leave as is
	pass