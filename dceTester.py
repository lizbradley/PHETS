import sys
import time
import math

import numpy as np


from DCE.DCE import embed
from DCE.Utilities import wav_to_txt, batch_wav_to_txt
from DCE.Tools import plot_power_spectrum
from DCE.Plots import make_frame
from PubPlots import plot_dce_pub, plot_waveform_sec
from DCE.Movies import vary_tau, slide_window, compare_vary_tau, compare_multi

# slide_window, vary_tau, compare_vary_tau: tau is in samples not seconds
# compare_multi: tau is in seconds; has all options for tau and crop *****


set_test = 6		# set here or with command line argument



if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()


if test == 0:
	batch_wav_to_txt('datasets/time_series/piano_revisit/C134C/scale')


if test == 1:
	for i in xrange(10):
		print 'Hello'
		note = i + 10
		piano = 'C134C'
		vary_tau(
			'datasets/time_series/%s/%s-%s.txt' % (piano, str(note), piano),
			tau_lims=(.001, .005),
			tau_inc=.001,			# seconds
			embed_crop=(.5*i, .5*(i+1)),  	# aka window position, in seconds
			ds_rate=20
		)             		# downsample rate (takes every third sample)


if test == 2:
	for i in xrange(7):
		note = i*10+10
		piano = 'C134C'
		slide_window(
			'datasets/time_series/%s/%s-%s.txt' % (piano, str(note), piano),
			window_size=.05*(i+1),    # seconds
			ds_rate=1,
			tau=.001,					# seconds
			window_step=1)      # how much to move window each frame


if test == 3:
	for i in xrange(7):
		note = (i+1)*10
		print 'note is %s ' % str(note)
		compare_vary_tau(
			'datasets/time_series/C135B/%s-C135B.txt' % str(note),
			'datasets/time_series/C134C/%s-C134C.txt' % str(note),
			tau_lims=(.001, .005),
			tau_inc=.001, 	# seconds
			embed_crop=(.5, .7),
			ds_rate=5
		)
		print 'note is still %s ' % str(note)


if test == 4:
	vary_tau(
		'datasets/time_series/C134C/49-C134C.txt',
		'output/DCE/test_4.mp4',
		tau_lims=(.001, .008),
		tau_inc=.001,  # seconds
		embed_crop=(1, 2),  # aka window position, in seconds
		ds_rate=1,
	)  # downsample rate (takes every third sample)


if test == 5:
	slide_window(
		'datasets/time_series/C134C/49-C134C.txt',
		'output/DCE/test_5.mp4',
		window_size=.1,    	# seconds
		tau=.001,			# seconds
		window_step=1,      	# how much to move window each frame
	)



if test == 6:
	compare_vary_tau(
		'datasets/time_series/C135B/49-C135B.txt',
		'datasets/time_series/C134C/49-C134C.txt',
		'output/DCE/test_6.mp4',
		tau_lims=(.001, .005),
		tau_inc=.001, 			 # seconds
		embed_crop=(.5, .7),
		ds_rate=5
	)




if test == 8:
	# still trying to figure out exactly how the units should work here
	plot_power_spectrum(
		'datasets/time_series/C134C/34-C134C.txt',
		'output/DCE/power_spectrum_34-C134C.png',
		crop=(1, 2),    # window for analysis (seconds)
	)



if test == 9:
	dir1, base1 = 'datasets/time_series/C134C', '-C134C.txt'
	dir2, base2 = 'datasets/time_series/viol', '-viol.txt'
	out_filename = 'output/DCE/viol_test_7_tau.25T.mp4'

	compare_multi(
		dir1, base1,
		dir2, base2,
		out_filename,

		i_lims=(36, 64), 		 # specify note range

		embed_crop_1='auto',	 # seconds or 'auto'
		embed_crop_2=(2, 2.3),	 # seconds or 'auto'
		auto_crop_length=.3,  	 	 # seconds for when embed_crop = 'auto'

		tau_1='auto detect',  	 # seconds 'auto detect' or 'auto ideal'. NOTE: 'auto detect' is considerably slower that 'auto ideal'
		tau_2='auto ideal',
		tau_T=math.pi, 		 	 # for auto tau. tau = period * tau_T

		save_trajectories=True,		 # to output/DCE/trajectories
		save_movie=True,			 # False for faster worm creation

		ds_rate=1


		# As of now tau cannot be specified for file1 and file 2 seperately. Let me know if this functionality is needed.

	)



if test == 10:
	
	dir1, base1 = 'datasets/time_series/C134C', '-C134C.txt'  # numerical index goes in front of base
	dir2, base2 = "datasets/time_series/C135B", '-C135B.txt'
	out_filename = 'output/DCE/C134vC135.mp4'
	
	compare_multi(
		dir1, base1,
		dir2, base2,
		out_filename,

		i_lims=(40, 45), 		 # specify note range

		embed_crop_1=(1, 1.3),	 # seconds or 'auto'
		embed_crop_2=(1, 1.3),	 # seconds or 'auto'
		auto_crop_length=.05,  	 # seconds for when embed_crop = 'auto'

		tau_1= .001,  			 # seconds 'auto detect' or 'auto ideal'. note 'auto detect' is considerably slower that 'auto ideal'
		tau_2= .001,  			 # seconds 'auto detect' or 'auto ideal'. note 'auto detect' is considerably slower that 'auto ideal'
		tau_T=math.pi, 		 	 # for auto tau. tau = period * tau_T

		m=2,

		save_trajectories=True,	 # to output/DCE/trajectories
		save_movie=True			 # False for faster worm creation
		
	)


if test == 11:
	############# IDA PAPER FIG 1 (using auto crop) ###############


	dir1, base1 = 'datasets/time_series/C134C', '-C134C.txt'  # numerical index goes in front of base
	dir2, base2 = "datasets/time_series/C135B", '-C135B.txt'
	out_filename = 'output/DCE/C134vC135_fontsize.mp4'

	compare_multi(
		dir1, base1,
		dir2, base2,
		out_filename,

		i_lims=(49, 50), 			 	# specify note range

		embed_crop_1='auto',		 	# seconds or 'auto'
		embed_crop_2='auto',  			# seconds or 'auto'
		auto_crop_length=.05, 			# seconds for when embed_crop = 'auto'

		tau_1=.01192, 			# seconds 'auto detect' or 'auto ideal'. note 'auto detect' is considerably slower that 'auto ideal'
		tau_2=.01192,
		tau_T=math.pi,  				# for auto tau. tau = period * tau_T

		normalize_volume=True,

		save_trajectories=True,  		# to output/DCE/trajectories
		save_movie=True,

		waveform_zoom=out_filename

	)


	plot_waveform_sec(

		'datasets/time_series/C135B/49-C135B.txt',		# in filename
		'output/DCE/time_series_zoom.png',				# out filename

		embed_crop='auto',
		auto_crop_length=.05,
	)


if test == 12:
	############# IDA PAPER FIG 1 (explicit) ###############

	sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt')
	traj = embed(sig, tau=.01192, m=2, time_units='seconds', crop=(1.72132, 1.77132))
	np.savetxt('datasets/IDA_PAPER/49-C135B.txt', traj)
	plot_dce_pub(traj, 'output/DCE/testing.png')

print("time elapsed: %d seconds" % (time.time() - start_time))

# PARAMETER TEST CASES 5/3/17 #
#################################################################
#  tau: auto ideal		crop: explicit	  #             #
#  tau: auto ideal		crop: auto        #             #
#  tau: auto detect		crop: explicit    #             #
#  tau: auto detect		crop: auto        #             #
#  tau: explicit		crop: explicit    #             #
#  tau: explicit		crop: auto        #             #
#################################################################