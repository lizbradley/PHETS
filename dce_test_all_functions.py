import numpy as np
from DCE.Movies import compare_multi
from DCE.MovieTools import frames_to_movie


dir1, base1 = 'datasets/time_series/C134C', '-C134C.txt'  # numerical index goes in front of base
dir2, base2 = "datasets/time_series/C135B", '-C135B.txt'

compare_multi(

	dir1, base1,
	dir2, base2,

	i_lims=(40, 50),  		# specify note range

	embed_crop_1='auto',  	# 'auto' or (start, stop) in seconds
	embed_crop_2=(2, 2.1),
	crop_auto_len=.1,  	# in seconds, for when embed_crop = 'auto'

	tau_1='auto ideal',		# in seconds or 'auto detect' or 'auto ideal'
	tau_2='auto detect', 	# note 'auto detect' is considerably slower that 'auto ideal'
	tau_T=np.pi,  			# for auto tau. tau = period * tau_T

	normalize_volume=True,

	save_worms=True,  		# save embedded worms as text to output/DCE/saved_worms
	save_movie=True  		# skip printing frames for faster worm creation

)

frames_to_movie('output/DCE/C134vC135.mp4', framerate=1)


# TODO: auto tau broken
