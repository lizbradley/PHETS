## Synopsis

**P**ersistent **H**omology on **E**mbedded **T**ime-**S**eries

This module offers high-level tools for exploration and visualization of delay coordinate 
embedding and persistent homology. It is used to investigate the utilization of these tools together as a signal 
processing technique.

Also included is a dataset of time-series (mostly musical instrument recordings) and 
higher dimensional trajectories as .txt files.



## Installation

### Dependencies
* Python 2.7
* popt
* ffmpeg
* gnuplot (with pngcairo)


### Setup

#### Install Dependencies


###### Popt
```bash
wget http://rpm5.org/files/popt/popt-1.16.tar.gz
tar -xvzf popt-1.16.tar.gz
cd popt-1.16
./configure --prefix=/usr --disable-static &&
make
sudo make install
export LD_RUN_PATH="/usr/lib"
```

###### ffmpeg and gnuplot
```bash
sudo apt-get install ffmpeg
sudo apt-get install gnuplot-x11
```

#### Install PHETS

```bash
git clone https://github.com/eeshugerman/PHETS.git
cd PHETS
pip install -r requirements.txt
```



## Demo
```python

import numpy as np
from signals import TimeSeries
from utilities import idx_to_freq
from DCE.movies import slide_window
from PH import Filtration
from config import default_filtration_params as filt_params

ts = TimeSeries(
    'datasets/time_series/C135B/49-C135B.txt',
    crop=(0, 5),
    num_windows=250,
    window_length=.05,
    time_units='seconds'    # defaults to 'samples'
)

# the slide_window function will create a movie showing an embedding for each
# window of the time series and return the trajectory

tau = (1 / idx_to_freq(49)) / np.pi      # first, choose tau = period / pi 

traj = slide_window(
    ts,
    m=2, tau=tau,
    out_fname='output/demo/embed_movie.mp4'
)

```
![embed movie](docs/readme/embed_movie.gif "embed_movie.mp4")

```python
# alternatively, we could skip the movie and embed explicitly:
traj = ts.embed(m=2, tau=tau)

# now, lets build a filtration from the trajectory that is shown in the 100th 
# frame of the slide_window movie
traj_window = traj.windows[100]

# parameters used to build the filtration:
filt_params.update(
    {
        'ds_rate': 25,
        'num_divisions': 10,                # number of epsilon vals in filtration
        # 'max_filtration_param': .05,      # if > 0, explicit
        'max_filtration_param': -10,        # if < 0, stops st first 10 dim simplex
        # 'use_cliques': True,
    }
)

# build the filtration:
filt = Filtration(traj_window, filt_params)

filt.movie('output/demo/filt_movie.mp4')

```

![filtration movie](docs/readme/filt_movie.gif "filt_movie.mp4")


A filtration can be summarized by its homology, which may be expressed as a persistence rank function:

Or as a persistence rank function:
```python
filt.plot_PRF('output/demo/PRF.png')        # plot the persistence rank function
```

![perseistence rank function](docs/readme/PRF.png "PRF.png")

Persistence rank functions are amenable to statistical analysis. Several functions are provided for exploring these properties.
`PRFstats.plot_clusters()`, for example, takes two disjoint sets of samples ('training' and 'test') from each of two input signals,
computes the mean PRF for each training set, and plots the L2 distances from these means to the PRFs of the test sets. 
In the image below, two pianos are compared (left) and not easily distinguished; a viol and a piano are compared (right) and clustering is observed.

![not so different](docs/readme/clusters.png "left: piano vs piano | right: viol vs piano")

See `reference.pdf` for more information.

