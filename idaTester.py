import sys, time
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from ROC import PRF_vs_FFT_v2
from config import default_filtration_params as filt_params
from DCE.DCE import embed
from PH import Filtration, make_movie, load_saved_filtration
from PubPlots import plot_filtration_pub, plot_PD_pub, plot_waveform_sec, plot_dce_pub
from Tools import normalize_volume, sec_to_samp

set_test = 0			# set test number here or with command line argument


if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()




piano_sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt')
piano_sig = normalize_volume(piano_sig)
# piano_traj = embed(piano_sig, tau=.01192, m=2, time_units='seconds', crop=(1.72132, 1.77132))		# current

# piano_traj = embed(piano_sig, tau=.01192, m=2, time_units='seconds', crop=(1.72132, 1.77872))		# experiment 2000 wits
# piano_traj = embed(piano_sig, tau=.001216, m=2, time_units='seconds', crop=(1.72132, 1.77152))	# experiment small tau

# paper_path = '/home/elliott/programming/phets_notes/IDA 2017/figs/'
# paper_path = '/home/elliott/programming/phets_notes/IDA 2017/figs/experiment_2000_wits/'
# paper_path = '/home/elliott/programming/phets_notes/IDA 2017/figs/experiment_small_tau/'
ticks = [-.06, -.02, .02, .06]

#
# if test == 1:
# 	# figure 1a
#
# 	plot_waveform_sec(
# 		'datasets/IDA_PAPER/49-C135B.txt',
# 		paper_path + 'Figure1/49_C135B_time_series.png',
# 		crop=(1.72132, 1.77132),
# 	)
#
#
#
# if test == 2:
# 	# figure 1b
#
# 	plot_dce_pub(
# 		'datasets/IDA_PAPER/49-C135B_embedded.txt',
# 		paper_path + 'Figure1/49_C135B_embedded_taudetect_525.png'
# 	)
#
#
#
# if test == 3:
# 	# figure 2a
# 	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
# 	
# 	filt_params.update(
# 		{
# 			'ds_rate': 1,
# 			'worm_length': 2000,
# 			'min_filtration_param': .001,
# 			'max_filtration_param': .005,
# 			'num_divisions': 2,
# 			'use_cliques': False
# 		})
#
# 	# filtration = Filtration(in_filename, filt_params)
# 	filtration = load_saved_filtration()
#
# 	#
# 	# make_movie(
# 	# 	filtration,
# 	# 	"output/IDA_PAPER/49-C135B.mp4",
# 	# 	color_scheme='none',
# 	# )
#
# 	plot_filtration_pub(
# 		filtration, 2,
# 		'output/IDA_PAPER/fig2a.png',
# 		landmark_size=3,
# 		line_width=.3,
# 		show_eps=False,
# 		label='(a)'
# 	)
# #
# 	# plot_PD_pub(filtration, 'output/IDA_PAPER/fig3_PD.png')
#
#
#
#
# if test == 4:
# 	# figure 2b
#
# 	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
# 	
# 	filt_params.update(
# 		{
# 			'ds_rate': 10,
# 			'worm_length': 2000,
# 			'min_filtration_param': .001,
# 			'max_filtration_param': .005,
# 			'num_divisions': 2,
# 			'use_cliques': False
# 		})
#
# 	filtration = Filtration(in_filename, filt_params)
# 	# filtration = load_saved_filtration()
#
#
#
# 	plot_filtration_pub(
# 		filtration, 1,
# 		'output/IDA_PAPER/fig2b.png',
# 		landmark_size=10,
# 		show_eps=False,
# 		label='(b)'
# 	)
#
#
# if test == 5:
# 	# figure 2c
#
#
# 	
# 	filt_params.update(
# 		{
# 			'ds_rate': 40,
# 			'worm_length': 2000,
# 			'min_filtration_param': .001,
# 			'max_filtration_param': .005,
# 			'num_divisions': 2,
# 			'use_cliques': False
# 		})
#
# 	filtration = Filtration('datasets/IDA_PAPER/49-C135B_embedded.txt', filt_params)
# 	# filtration = load_saved_filtration()
#
#
#
# 	plot_filtration_pub(
# 		filtration, 2,
# 		'output/IDA_PAPER/fig2c.png',
# 		landmark_size=10,
# 		show_eps=False,
# 		label='(c)')
#

# test = 10

# test = 10

if test == 10:

	# figure 1a, 1b
	# add time (s) label
	fig = plt.figure(figsize=(10, 3.5), tight_layout=True, dpi=600)

	gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

	ax1 = fig.add_subplot(gs[0])
	ax2 = fig.add_subplot(gs[1], sharey=ax1)

	plot_waveform_sec(
		ax1,
		piano_sig,
		crop=(1.72132, 1.77132),
		label='(a)',
		normalize=True,
		yticks=ticks
	)


	plot_dce_pub(
		ax2,
		piano_traj,
		ticks=ticks,
		label='(b)'
	)

	plt.savefig(paper_path + 'fig1.png')


# test = 20

# crop = sec_to_samp((1, 3))

if test == 20:
	# figure 2a, 2b, 2c

	fig = plt.figure(figsize=(10, 3.5), tight_layout=True, dpi=700)
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)


	# 2a #
	
	filt_params.update(
		{
			'ds_rate': 1,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False
		})

	filtration = Filtration(piano_traj, filt_params)

	plot_filtration_pub(
		filtration, 2,
		ax1,
		landmark_size=3,
		line_width=.3,
		show_eps=False,
		label='(a)',
		ticks=ticks
	)


	# 2b #
	
	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False,
		})

	filtration = Filtration(piano_traj, filt_params)

	plot_filtration_pub(
		filtration, 2,
		ax2,
		landmark_size=5,
		show_eps=False,
		label='(b)',
		ticks=ticks,

	)

	# 2c #
	
	filt_params.update(
		{
			'ds_rate': 40,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .005,
			'num_divisions': 2,
			'use_cliques': False,
		})

	filtration = Filtration(piano_traj, filt_params)

	plot_filtration_pub(
		filtration, 2,
		ax3,
		landmark_size=10,
		show_eps=False,
		label='(c)',
		ticks=ticks,
	)


	plt.savefig(paper_path + 'fig2.png')


# test = 30

if test == 30:
	# figures 3a, 3b, 3c, 3d
	
	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 2000,
			'min_filtration_param': .000,
			'max_filtration_param': .010,
			'num_divisions': 20,
			'use_cliques': False,
			# 'use_cliques': True,
		})

	filtration = Filtration(piano_traj, filt_params)

	fig = plt.figure(figsize=(8.5, 7.5), tight_layout=True, dpi=700)
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)

	plot_filtration_pub(
		filtration, 1, ax1,
		label='(a) $\epsilon = 0.001$',
		show_eps=False,
		ticks=ticks,
	)

	plot_filtration_pub(
		filtration, 3, ax2,
		label='(b) $\epsilon = 0.002$',
		show_eps=False,
		ticks=ticks
	)

	plot_filtration_pub(
		filtration, 5, ax3,
		label='(c) $\epsilon = 0.003$',
		show_eps=False,
		ticks=ticks
	)

	# make_movie(filtration, paper_path + 'fig3movie.mp4')

	plot_PD_pub(filtration, ax4, label='(d)')

	plt.savefig(paper_path + 'fig3.png')

# test = 40

if test == 40:

	# figures 4a, 4b, 4c, 4d

	fig = plt.figure(figsize=(8.5, 7.5), tight_layout=True, dpi=700)
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)

	# CLARINET #

	sig = np.loadtxt('datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt')
	traj = embed(sig, tau=32, m=2, time_units='samples', crop=(100000, 102205), normalize_crop=True)

	
	filt_params.update(
		{
			'ds_rate': 20,
			'worm_length': 2000,
			'min_filtration_param': 0,
			'max_filtration_param': -15,
			'num_divisions': 10,

		})

	filtration = Filtration(traj, filt_params)


	plot_filtration_pub(filtration, 8, ax1)						# 4a
	plot_PD_pub(filtration, ax3)									# 4c


	# VIOL #

	sig = np.loadtxt('datasets/time_series/viol/40-viol.txt')
	traj = embed(sig, tau=32, m=2, time_units='samples', crop=(50000, 52205), normalize_crop=True)

	
	filt_params.update(
		{
			'ds_rate': 20,
			'worm_length': 2000,
			'min_filtration_param': 0,
			'max_filtration_param': -15,
			'num_divisions': 10,

		})

	filtration = Filtration(traj, filt_params)
	plot_filtration_pub(filtration, 4, ax2)							# 4b
	plot_PD_pub(filtration, ax4)									# 4d

	plt.savefig(paper_path + 'fig4.png')


test = 50




###### new Sunday

if test == 50:
	filt_params.update(
		{
			'max_filtration_param': -15,
			'num_divisions': 10,
			# 'use_cliques': True,
		}
	)

	PRF_vs_FFT_v2(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		'datasets/time_series/viol/40-viol.txt',
		'output/ROC/test_26_scvar.png',

		'clarinet',
		'viol',

		crop_1=(75000, 180000),
		crop_2=(35000, 140000),

		tau=32,  # samples
		m=2,

		window_length=2033,
		num_windows=50,
		num_landmarks=100,
		FT_bins=50,
		k=(0, 5, .01),  # min, max, int
		load_saved_filts=True,
		normalize_volume=True

	)
	
	
if test == 60:
	# figure 6a, PD upright piano

	sig = np.loadtxt('datasets/IDA_PAPER/40-C144F.txt')
	traj = embed(sig, tau=4, m=2, time_units='samples', crop=(70000, 72205), normalize=True)

	
	filt_params.update(
		{
			'ds_rate': 20,
			'worm_length': 2000,
			'min_filtration_param': 0,
			'max_filtration_param': -15,
			'num_divisions': 10,

		})

	filtration = Filtration(traj, filt_params)

	plot_PD_pub(filtration, paper_path + 'fig6a.png')

# test = 61
if test == 61:
	# figure 6b, PD grand piano

	sig = np.loadtxt('datasets/IDA_PAPER/40-C134C.txt')
	traj = embed(sig, tau=4, m=2, time_units='samples', crop=(86000, 88205), normalize=True)

	
	filt_params.update(
		{
			'ds_rate': 20,
			'worm_length': 2000,
			'min_filtration_param': 0,
			'max_filtration_param': -15,
			'num_divisions': 10,

		})

	filtration = Filtration(traj, filt_params)
	# filtration = load_saved_filtration()

	plot_PD_pub(filtration, paper_path + 'fig6b.png')
