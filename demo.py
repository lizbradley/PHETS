import numpy as np
from Tools import idx_to_freq
from DCE import embed, plot_signal
from DCE.Movies import slide_window, vary_tau
from PH import Filtration, make_movie, make_PD
from config import default_filtration_params as filt_params


time_units = 'seconds'
crop = (1, 2)
tau = (1 / idx_to_freq(49)) / np.pi
m = 2

sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt')

plot_signal('output/demo/signal.png', sig, crop=crop, time_units=time_units)

trajs = slide_window(
	sig,
	'output/demo/embed_movie.mp4',
	tau=tau,
	m=2,
	window_size=.05,
	window_step=.1,
	crop=crop
)

traj = trajs[5]		# take embedding from 5th frame of movie

# traj = embed(sig, tau, m, crop=crop, time_units=time_units)		# alternatively, embed explicitly

filt_params.update(
	{
		'worm_length': 2000,
		'ds_rate': 50,
		'num_divisions': 10,
		# 'max_filtration_param': .05,
		'max_filtration_param': -10,
		'use_cliques': True,


	}
)

filt = Filtration(traj, filt_params)

make_movie(filt, 'output/demo/filt_movie.mp4')
make_PD(filt, 'output/demo/PD.png')