import numpy as np

from boilerplate import change_dir
change_dir()

from signals import TimeSeries
from embed.movies import slide_window
from PH import Filtration
from config import default_filtration_params as filt_params

# first, format data file
# fname = 'datasets/time_series/WAIS_age_dD_d18O_xs.txt'
fname = '../WAIS_age_dD_d18O_xs.txt'
# data = np.loadtxt(fname, skiprows=1, delimiter=',')
# data = data[:, 2]                           # select column
fname = '{}.clean'.format(fname)              # new fname
# np.savetxt(fname, data)                     # save cleaned/formatted data

# initialize a TimeSeries object from formatted data file
ts = TimeSeries(
	fname,
    crop=(20000, 40000),
	num_windows=10,
	window_length=1000
)

# visual representation of window and crop geometry
# ts.plot('output/liz/signal.png')          # full signal
ts.plot_crop('output/liz/signal_crop.png')  # cropped section only

# make embed movie from the TimeSeries, returning a Trajectory
slide_window(ts, 'output/liz/embed_movie.mp4', m=2, tau=200)

# alternatively, we can embed explicitly (much faster than making the movie)
traj = ts.embed(m=2, tau=200)


traj_window = traj.windows[3]          # take embedding from 3rd window (ie 3rd frame of movie)

# parameters used to build the filtration:
filt_params.update(
	{
		'ds_rate': 10,
		'num_divisions': 15,                # number of epsilon vals in filtration
		# 'max_filtration_param': .05,      # if positive, explicit
		'max_filtration_param': -5,         # if negative, cuts off filtration when finds a 5 dim simplex
		'use_cliques': True,

	}
)

# initialize Filtration object from the Trajectory:
filt = Filtration(traj_window, filt_params)

filt.movie('output/liz/filt_movie.mp4')     # save the filtration/complexes movie
filt.plot_pd('output/liz/PD.png')           # save the persistence diagram
filt.plot_prf('output/liz/PRF.png')         # save the persistence rank function
