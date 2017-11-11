from boilerplate import change_dir
change_dir()

from signals import TimeSeries
import numpy as np
from utilities import idx_to_freq
from DCE.movies import slide_window
from PH import Filtration
from config import default_filtration_params as filt_params


# the following vars are passed the functions below, defined here for convenience

time_units = 'seconds'					# 'seconds' or 'samples'
crop = (0, 5)				 			# range of the signal that you want to play with
tau = (1 / idx_to_freq(49)) / np.pi		# embedding delay
m = 2 									# embedding dimension

print 'loading data...'                 # set file to work with below. use skiprows=1 to skip a one line header.
sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt', skiprows=0)

ts = TimeSeries(
    'datasets/time_series/C135B/49-C135B.txt',
    crop=(0, 5),
    num_windows=10,
    window_length=.05,
    time_units='seconds'

)

ts.plot('output/demo/timeseries.png')
traj = ts.embed(tau=(1 / idx_to_freq(49)) / np.pi, m=2)



# the following function creates a movie of the embeddings over a sliding window
slide_window(traj, 'output/demo/embed_movie.mp4')

#
# window = traj.windows()[5]

# # parameters used to build the filtration:
# filt_params.update(
#     {
#         'ds_rate': 25,
#         'num_divisions': 25,                # number of epsilon vals in filtration
#         # 'max_filtration_param': .05,      # if positive, explicit
#         'max_filtration_param': -10,        # if negative, cuts off filtration when finds a 10 dim simplex
#         'use_cliques': True,
#
#     }
# )
#
# # build the filtration:
# filt = Filtration(traj, filt_params, name='piano demo')
#
# make_movie(filt, 'output/demo/filt_movie.mp4')
# make_PD(filt, 'output/demo/PD.png')         # plot the persistence diagram
# make_PRF_plot(filt, 'output/demo/PRF.png')
