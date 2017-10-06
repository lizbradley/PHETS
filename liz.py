import numpy as np
from Tools import idx_to_freq
from DCE import embed, plot_signal
from DCE.Plots import plot_dce
from DCE.Movies import slide_window, vary_tau
from PH import Filtration, make_movie, make_PD, make_PRF_plot
from config import default_filtration_params as filt_params

# the following vars are passed the functions below, defined here for convenience

time_units = 'samples'				# 'seconds' or 'samples'
crop = (20000, 40000)			        # range of the signal that you want to play with
tau = 200                                       #embedding delay
m = 2 						# embedding dimension

print 'loading data...'
# load data from text file. you can use skiprows=1 to skip a header.
sig = np.loadtxt('datasets/time_series/WAIS_age_dD_d18O_xs.txt',
                 skiprows=1,
                 delimiter=','   # takes whitespace by default
)

sig = sig[:,2]
# see signal.png for a plot of the full signal and region that will be cropped
plot_signal('output/liz/signal.png', sig, window=crop, time_units=time_units)

# the following function creates a movie of the embeddings over a sliding window
# returns a list of the embeddings (trajectories), one for each window
# can comment out these lines if you load the data into trajs from a file
# using the lines below.
# this code puts the frames in PHETS/DCE/frames if you want them
trajs = slide_window(
	sig,
	'output/liz/embed_movie.mp4',
	tau=tau,
	m=m,
	window_size=2000,
	window_step=2000,
	crop=crop,
)
# ...down to here

np.save('output/liz/trajs.npy', trajs)         # lets save trajs to file, so we don't have to call slide_window() every time we want to do PH on the data
                                                # after the inital run, the preceeding lines "trajs = slide_window(" to "np.save(...)" can be commented out 
trajs = np.load('output/liz/trajs.npy')

traj = trajs[3]		                        # take embedding from 3rd window of movie

# traj = embed(sig, tau, m, crop=crop, time_units=time_units)		# alternatively, embed explicitly


# parameters used to build the filtration:
filt_params.update(
	{
		'ds_rate': 25,
		'num_divisions': 15,                # number of epsilon vals in filtration
		# 'max_filtration_param': .05,      # if positive, explicit
		'max_filtration_param': -5,        # if negative, cuts off filtration when finds a 5 dim simplex
		'use_cliques': True,

	}
)

# build the filtration:
# filt = Filtration(traj, filt_params, title='piano demo')
#
# make_movie(filt, 'output/liz/filt_movie.mp4')
# make_PD(filt, 'output/liz/PD.png')  # make the persistence diagram
# make_PRF_plot(filt, 'output/liz/PRF.png')
