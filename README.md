## Synopsis

**P**ersistent **H**omology on **E**mbedded **T**ime-**S**eries

This repository offers high-level tools for exploration and visualization of delay coordinate 
embedding and persistent homology and is used to investigate the utilization of these tools together as a signal 
processing technique.

Also included is a dataset of time-series (mostly musical instrument recordings) and 
higher dimensional trajectories in .txt.



## Installation

### Dependencies
* Python 2.7
* GCC 5
* popt.h
* ffmpeg
* gnuplot


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
sudo apt-get install gnuplot
```

#### Install PHETS

```bash
git clone https://github.com/eeshugerman/PHETS.git
cd PHETS
pip install -r requirements.txt
```



## Usage
```python
import numpy as np
from Tools import idx_to_freq
from DCE.Movies import slide_window, vary_tau

time_units = 'seconds'					# 'seconds' or 'samples'
crop = (1, 3)						 	# range of the signal that you want to play with
tau = (1 / idx_to_freq(49)) / np.pi		# embedding delay
m = 2 									# embedding dimension

sig = np.loadtxt('datasets/time_series/C135B/49-C135B.txt')

trajs = slide_window(
	sig,
	'output/demo/embed_movie.mp4',
	tau=tau,
	m=m,
	window_size=.05,  			# this is in seconds
	window_step=.05,
	crop=crop,
	title='49-C135B.txt'		# optional, for labelling. will be obsolete when Signal class is implemented

)
```
![well actually this is a gif](docs/readme/embed_movie.gif "embed_movie.mp4")




See demo.py and reference.pdf. 

