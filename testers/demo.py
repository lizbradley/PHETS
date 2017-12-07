from boilerplate import change_dir
change_dir()

import numpy as np
from signals import TimeSeries
from utilities import idx_to_freq
from embed.movies import slide_window
from phomology import Filtration, load_filtration
from PRFstats import plot_ROCs
from config import default_filtration_params as filt_params

# first load data into a TimeSeries instance. we have the choice of specifying
# a crop (start, stop), slicing into windows (evenly spaced), and working in
# samples or seconds

ts = TimeSeries(
    'datasets/time_series/C135B/49-C135B.txt',
    crop=(0, 5),
    num_windows=250,
    window_length=.05,
    time_units='seconds'    # defaults to 'samples'
)

# the slide_window function will create a movie showing an embedding for each
# window of the time series
tau = (1 / idx_to_freq(49)) / np.pi      # choose tau = period / e

slide_window(ts, 'output/demo/embed_movie.mp4', m=2, tau=tau)

# alternatively, we could skip the movie and embed explicitly:
traj = ts.embed(m=2, tau=tau)

# now, lets build a filtration from the trajectory that is shown in the fifth
# frame of the slide_window movie
traj_window = traj.windows[100]

# parameters used to build the filtration:
filt_params.update(
    {
        'ds_rate': 25,
        'num_divisions': 50,                 # number of epsilon vals in filtration
        'max_filtration_param': .02,         # if > 0, explicit
        # 'max_filtration_param': -5,        # if < 0, stops st first 10 dim simplex
    }
)

# build the filtration:
filt = Filtration(traj_window, filt_params)
filt.movie(
    'output/demo/filt_movie.mp4',
    alpha=.5,
    color_scheme='highlight new'
)
filt.plot_pd('output/demo/PD.png')          # plot the persistence diagram
filt.plot_prf('output/demo/PRF.png')        # plot the persistence rank function




ts1 = TimeSeries(
    'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
    crop=(75000, 180000),
    num_windows=50,
    window_length=1500,
    vol_norm=(0, 0, 1)  # (full, crop, windows)
)


ts2 = TimeSeries(
    'datasets/time_series/viol/40-viol.txt',
    crop=(35000, 140000),
    num_windows=50,
    window_length=1500,
    vol_norm=(0, 0, 1)
)

ts1.plot('output/PRFstats/ts1.png')
ts2.plot('output/PRFstats/ts2.png')

traj1 = ts1.embed(tau=32, m=2)
traj2 = ts2.embed(tau=32, m=2)


filt_params.update({
    'max_filtration_param': -21,
    'num_divisions': 20,
    'ds_rate': 20
})

plot_ROCs(
    traj1, traj2,
    'output/demo/ROCs.png',
    filt_params,
    k=(0, 10.01, .01)
)
