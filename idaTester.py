import sys, time
from config import default_filtration_params as parameter_set
from PH import Filtration, make_movie, load_saved_filtration
from PubPlots import plot_filtration_pub, plot_PD_pub, plot_waveform_sec, plot_dce_pub

set_test = 1		# set test number here or with command line argument


if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()


# from DCE.DCE import embed
# sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt')
# traj = embed(sig, tau=.01192, m=2, time_units='seconds', embed_crop=(1.72132, 1.77132), normalize=True)
# np.savetxt('datasets/IDA_PAPER/49-C135B_embedded.txt', traj)


if test == 1:
	# figure 1a

	plot_waveform_sec(
		'datasets/IDA_PAPER/49-C135B.txt',
		'output/IDA_PAPER/fig1a.png',
		crop=(1.72, 1.77)
	)


if test == 2:
	# figure 1b

	plot_dce_pub(
		'datasets/IDA_PAPER/49-C135B.txt',
		'output/IDA_PAPER/fig2b.png'
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

	# filtration = Filtration(in_filename, filt_params)
	filtration = load_saved_filtration()

	#
	# make_movie(
	# 	filtration,
	# 	"output/IDA_PAPER/49-C135B.mp4",
	# 	color_scheme='none',
	# )

	plot_filtration_pub(filtration, 2, 'output/IDA_PAPER/fig2a.png',
						landmark_size=1)
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


	make_movie(
		filtration,
		"output/IDA_PAPER/49-C135B.mp4",
		color_scheme='none',
	)

	plot_filtration_pub(filtration, 1, 'output/IDA_PAPER/fig2b.png')


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
	# filtration = Filtration(traj, filt_params)
	# filtration = load_saved_filtration()


	make_movie(
		filtration,
		"output/IDA_PAPER/49-C135B.mp4",
		alpha=.5
	)

	plot_filtration_pub(filtration, 2, 'output/IDA_PAPER/fig2c.png')




if test == 6:
	# figure 3a, 3b, 3c, 3d
	in_filename = 'datasets/IDA_PAPER/49-C135B_embedded.txt'
	filt_params = parameter_set
	filt_params.update(
		{
			'ds_rate': 10,
			'worm_length': 2000,
			'min_filtration_param': .001,
			'max_filtration_param': .011,
			'num_divisions': 20,
			'use_cliques': False
		})

	filtration = Filtration(in_filename, filt_params)
	# filtration = load_saved_filtration()


	# make_movie(
	# 	filtration,
	# 	"output/IDA_PAPER/49-C135B.mp4",
	# 	color_scheme='none',
	# )


	plot_filtration_pub(filtration, 1, 'output/IDA_PAPER/fig3a.png')
	plot_filtration_pub(filtration, 3, 'output/IDA_PAPER/fig3b.png')
	plot_filtration_pub(filtration, 5, 'output/IDA_PAPER/fig3c.png')
	plot_PD_pub(filtration, 'output/IDA_PAPER/fig3d.png')



if test == 7:
	# figure 4a
	pass


if test == 8:
	# figure 4b
	pass


if test == 9:
	# figure 5a
	# leave as is
	pass

if test == 10:
	# figure 5b
	# leave as is
	pass
