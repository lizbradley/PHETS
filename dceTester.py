import sys
import time
import math
from DCE.DCETools import wav_to_txt, batch_wav_to_txt, plot_power_spectrum
from DCE.DCEPlotter import make_window_frame
from DCE.DCEMovies_helper import frames_to_movie
from DCE.DCEMovies import vary_tau, slide_window, compare_multi, compare_vary_tau, compare_multi_auto_tau, compare_multi_auto


# TODO: test/debug compare_multi_auto(), remove compare_multi_auto_tau() and compare_multi()
# TODO: embed() wrapper
# TODO: fix errors on Nikki's computer
# TODO: bump up precision of requency detection


test = 9
# test = int(sys.argv[1])

print 'running test %d...' % test

start_time = time.time()

if test == 0:
	batch_wav_to_txt('input\piano_data\C134C')
	batch_wav_to_txt('input\piano_data\C135B')


if test == 1:
	for i in xrange(10):
		print 'Hello'
		note = i + 10
		piano = 'C134C'
		vary_tau(
			'datasets/time_series/%s/%s-%s.txt' % (piano, str(note), piano),
			tau_lims=(1, 100),
			tau_inc=5,
			embed_crop=(.5*i, .5*(i+1)),  	# aka window position, in seconds
			ds_rate=20
		)             		# downsample rate (takes every third sample)
		print 'hi'
		frames_to_movie('output/DCE/vary_tau_%s_%s.mp4' % (str(note), piano), framerate=1)


if test == 2:
	for i in xrange(7):
		note = i*10+10
		piano = 'C134C'
		slide_window(
			'datasets/time_series/%s/%s-%s.txt' % (piano, str(note), piano),
			window_size=.05*(i+1),    # seconds
			ds_rate=1,
			tau=10,
			step_size=1)      # how much to move window each frame
		frames_to_movie('output/DCE/slide_window_scale_tau_%s_%s.mp4' % (str(note), piano), framerate=1)


if test == 3:
	for i in xrange(7):
		note = (i+1)*10
		print 'note is %s ' % str(note)
		compare_vary_tau(
			'datasets/time_series/C135B/%s-C135B.txt' % str(note),
			'datasets/time_series/C134C/%s-C134C.txt' % str(note),
			tau_lims=(1, 40),
			tau_inc=2,
			embed_crop=(.5, .7),
			ds_rate=5
		)
		print 'note is still %s ' % str(note)
		frames_to_movie('output/DCE/compare_tau_%s.mp4' % str(note), framerate=1)


if test == 4:
	dir1 = "datasets/time_series/C134C"
	dir2 = "datasets/time_series/C135B"
	tau = 10
	compare_multi(
		dir1, '-C134C.txt',
		dir2, '-C135B.txt',
		tau,
		embed_crop=(.5, .7))
	frames_to_movie('output/DCE/compare_multi_test.mp4', framerate=1)


if test == 5:
	dir1 = "datasets/time_series/C134C"
	dir2 = "datasets/time_series/C135B"
	tau_T = 1/math.pi

	compare_multi_auto_tau(
		dir1, '-C134C.txt',
		dir2, '-C135B.txt',
		tau_T,
		embed_crop=(1, 1.1),
		ds_rate=5,
		i_lims=(20, 35)
	)

	frames_to_movie('output/DCE/compare_auto_tau__1_pi_t5.mp4', framerate=1)


if test == 7:
	dir1, base1 = 'datasets/time_series/C134C', '-C134C.txt'
	dir2, base2 = "datasets/time_series/viol_data", '-viol.txt'

	window = (1, 1.5)
	tau_T = 1/math.pi     # set tau / period ratio

	compare_multi_auto_tau(
		dir1, base1,
		dir2, base2,
		tau_T,
		embed_crop=window,
		ds_rate=1,
		i_lims=(36, 64),  # specify note range
		dpi=250
	)

	frames_to_movie('output/DCE/viol_test_7_tau.25T.mp4', framerate=1)

if test == 8:
	# still trying to figure out exactly how the units should work here
	plot_power_spectrum(
		'datasets/time_series/C134C/34-C134C.txt',
		'output/DCE/power_spectrum_34-C134C.png',
		crop=(1, 2),    # window for analysis (seconds)
	)


if test == 9:
	dir1, base1 = 'datasets/time_series/C134C', '-C134C.txt'
	dir2, base2 = "datasets/time_series/viol", '-viol.txt'

	compare_multi_auto(
		dir1, base1,
		dir2, base2,

		i_lims=(36, 64), 		 # specify note range

		embed_crop_1='auto',	 # seconds or 'auto'
		embed_crop_2='auto',	 # seconds or 'auto'
		crop_auto_len=.3,  	 	 # seconds for when embed_crop = 'auto'

		tau=.001134,			 # seconds 'auto detect' or 'auto ideal'
		tau_T=1 / math.pi, 		 # tau_sec = period * tau_T

		save_worms=True,		 # to output/DCE/saved_worms
		save_movie=True			 # False for faster worm creation


		# TESTS #
		#################################################################
		#  tau: auto ideal		crop: explicit	  #  PASSED             #
		#  tau: auto ideal		crop: auto        #  IndexError         #
		#  tau: auto detect		crop: explicit    #  ValueError         #
		#  tau: auto detect		crop: auto        #  ZerroDivisionError #
		#  tau: explicit		crop: explicit    #  PASSED             #
		#  tau: explicit		crop: auto        #  IndexError         #
		#################################################################


	)

	frames_to_movie('output/DCE/viol_test_7_tau.25T.mp4', framerate=1)





print("time elapsed: %d seconds" % (time.time() - start_time))
