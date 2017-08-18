import numpy as np
from Tools import idx_to_freq
from DCE import embed, plot_signal
from DCE.Plots import plot_dce
from DCE.Movies import slide_window, vary_tau
from PH import Filtration, make_movie, make_PD
from config import default_filtration_params as filt_params


time_units = 'seconds'  # so you don't have to re-type
crop = (1, 2)  # range of the signal that you want to play with
tau = (1 / idx_to_freq(49)) / np.pi  # embedding delay
# could specify tau = 0.1 if wanted 0.1 sec (or 2 if wanted 2 sample intervals, but would then need to set time units to 'samples'
m = 2 # embedding dimension

# loads the data
sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt')

# the call, which includes setting where it puts the results
plot_signal('output/demo/signal.png', sig, crop=crop, time_units=time_units)
# following line does the right thing with axis labels

# trajs = slide_window(
# 	sig,
# 	'output/demo/embed_movie.mp4',
# 	tau=tau,
# 	m=2,
# 	window_size=.05,  # this is in seconds
# 	window_step=.1,
# 	crop=crop
# )
#
# traj = trajs[5]		# take embedding from 5th frame of movie

traj = embed(sig, tau, m, crop=crop, time_units=time_units)		# alternatively, embed explicitly

plot_dce()

# parameters used to build the filtration:
filt_params.update(
	{
		'worm_length': 2000,
		'ds_rate': 50,
		'num_divisions': 10,  # number of epsilon vals in filtration
		# 'max_filtration_param': .05,  # if positive, explicit;
		'max_filtration_param': -10,  # if negative, cuts off filtration when finds a 10 dim simplex
		'use_cliques': True,


	}
)

# build the filtration:
filt = Filtration(traj, filt_params)

make_movie(filt, 'output/demo/filt_movie.mp4')
make_PD(filt, 'output/demo/PD.png')  # make the persistence diagram
