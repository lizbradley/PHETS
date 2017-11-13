from boilerplate import change_dir
change_dir()

from signals import TimeSeries
import numpy as np
from utilities import idx_to_freq
from DCE.movies import slide_window
from PH import Filtration
from config import default_filtration_params as filt_params


ts = TimeSeries(
    'datasets/time_series/C135B/49-C135B.txt',
    crop=(0, 5),
    num_windows=10,
    window_length=.05,
    time_units='seconds'
)

ts.plot('output/demo/timeseries.png')

# the following function creates a movie of the embeddings over a sliding
# window and returns the Trajectory
traj = slide_window(
    ts,
    m=2, tau=(1 / idx_to_freq(49)) / np.pi,
    out_fname='output/demo/embed_movie.mp4'
)


# alternatively, we could skip the movie and embed explicitly:
# traj = ts.embed(m=2, tau = (1 / idx_to_freq(49)) / np.pi)

# now, lets build a filtration from the trajectory that is shown in the fifth
# frame of the slide_window movie
window = traj.windows[5]

# parameters used to build the filtration:
filt_params.update(
    {
        'ds_rate': 25,
        'num_divisions': 25,                # number of epsilon vals in filtration
        # 'max_filtration_param': .05,      # if > 0, explicit
        'max_filtration_param': -10,        # if < 0, stops st first 10 dim simplex
        'use_cliques': True,
    }
)

# build the filtration:
filt = Filtration(traj, filt_params)

filt.movie('output/demo/filt_movie.mp4')
filt.plot_PD('output/demo/PD.png')          # plot the persistence diagram
filt.plot_PRF('output/demo/PRF.png')        # plot the persistence rank function
