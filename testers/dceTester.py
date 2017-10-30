import time, math

from Utilities import wav_to_txt, batch_wav_to_txt, tester_boilerplate
from DCE.Movies import vary_tau, slide_window, compare_vary_tau, compare_multi

test, start_time = tester_boilerplate(set_test=4)


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
			crop=(.5*i, .5*(i+1)),  	# aka window position, in seconds
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
		tau_inc=.001,
		crop=(1, 1.1),
		time_units='seconds'
	)

if test == 5:
	vary_tau(
		'datasets/time_series/C134C/49-C134C.txt',
		'output/DCE/test_5.mp4',
		tau_lims=(1, 100),
		tau_inc=10,
		crop=(100, 5000),
		time_units='samples'
	)


if test == 6:
	slide_window(
		'datasets/time_series/C134C/49-C134C.txt',
		'output/DCE/test_6.mp4',
		window_size=.1,
		tau=.001,
		window_step=1,
		time_units='seconds'
	)



if test == 7:
	slide_window(
		'datasets/time_series/C134C/49-C134C.txt',
		'output/DCE/test_7.mp4',
		window_size=1000,
		tau=150,
		window_step=1000,
		time_units='samples'
	)



if test == 8:
	compare_vary_tau(
		'datasets/time_series/C135B/49-C135B.txt',
		'datasets/time_series/C134C/49-C134C.txt',
		'output/DCE/test_8.mp4',
		tau_lims=(.001, .005),
		tau_inc=.001, 			 # seconds
		embed_crop=(.5, .7),
		ds_rate=4,
	)






if test == 9:
	dir1, base1 = 'datasets/time_series/C134C', '-C134C.txt'
	dir2, base2 = 'datasets/time_series/viol', '-viol.txt'
	out_filename = 'output/DCE/test_9.mp4'

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

