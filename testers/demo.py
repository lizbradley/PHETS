from boilerplate import change_dir
change_dir()

import numpy as np
from signals import TimeSeries
from utilities import idx_to_freq
from DCE.movies import slide_window
from PH import Filtration
from config import default_filtration_params as filt_params

# first load data into a TimeSeries instance. we have the choice of specifying a crop
# (start, stop), slicing into windows (evenly spaced), and working in samples
# or seconds

ts = TimeSeries(
    'datasets/time_series/C135B/49-C135B.txt',
    crop=(0, 5),
    num_windows=250,
    window_length=.05,
    time_units='seconds'    # defaults to 'samples'
)

# the slide_window function will create a movie showing an embedding for each
# window of the time series
tau = (1 / idx_to_freq(49)) / np.e      # choose tau = period / e

# traj = slide_window(
#     ts,
#     m=2, tau=tau,
#     out_fname='output/demo/embed_movie.mp4'
# )

# alternatively, we could skip the movie and embed explicitly:
traj = ts.embed(m=2, tau=tau)

# now, lets build a filtration from the trajectory that is shown in the fifth
# frame of the slide_window movie
traj_window = traj.windows[100]

# parameters used to build the filtration:
filt_params.update(
    {
        'ds_rate': 25,
        'num_divisions': 20,                # number of epsilon vals in filtration
        # 'max_filtration_param': .05,      # if > 0, explicit
        'max_filtration_param': -20,        # if < 0, stops st first 10 dim simplex
        'use_cliques': True,
    }
)

# build the filtration:
filt = Filtration(traj_window, filt_params)

filt.movie('output/demo/filt_movie.mp4')
filt.plot_PD('output/demo/PD.png')          # plot the persistence diagram
filt.plot_PRF('output/demo/PRF.png')        # plot the persistence rank function
