## Synopsis

**P**ersistent **H**omology on **E**mbedded **T**ime-**S**eries

This repository offers high-level tools for exploration and visualization of
* delay coordinate embedding from 1D to 2D
* persistent homology in 2D

and is used to investigate the utilization of these tools together as a signal 
processing technique.

Also included is a dataset of time-series (mostly musical instrument recordings) and 
higher dimensional trajectories in .txt.


3D functionality is under construction.

## Installation

### Dependencies
* Python 2.7
* GCC 5
* popt.h
* ffmpeg


### Setup
```bash
git clone https://github.com/eeshugerman/PHETS.git
cd PHETS
pip install -r requirements.txt
```



## Usage/Reference

Presently, functionality is split into three modules. Each module offers a handful 
of related functions and is driven by its own tester file. Each tester contains many 
tests which have accumulated over time. Many older tests are not kept up to date 
and will no longer run, but are kept for record keeping purposes.

Tests may be run by either setting the test variable at the top of the tester script, 
or by uncommenting the following line (`test = int(sys.argv[1])
`) and providing the test number as a command line 
argument.


#### DCE: delay coordinate embedding

DCE provides three main functions which each generate a different type of movie:

* **vary_tau()**  
example: test 4  
takes one input file shows embedding over range of tau

* **slide_window()**  
example: test 5  
takes one input file, shows embedding over a range of windows

* **compare_vary_tau()**  
example: test 6  
like vary_tau(), but takes two input files and shows their embeddings side by side.

* **compare_multi()**  
example: test 9    
takes two directories of (eg one with piano notes, another with range of viol 
notes), and generates a movie over a range note indexes (pitch). Tau and crop may be 
set explicity or automatically.  
The `save_worms` option will save all embedded trajectories which are generated for 
the movie to text files in`output/DCE/saved_worms`. These may then be passed as input
to the functions in `PH`.


#### PH: persistent homology
examples: test 16
* **make_movie()**    
make filtration movie

* **make_PD()**   
make persistence diagram

* **make_PRF_plot()**  
plot persistent rank function

#### PRFC: persistent rank function comparision


* **PRF_dist_plot()**  
example: test 4  
takes range of time-series files and a reference file. Generates PRF for each, and 
finds distances to reference PRF,
plots distance vs index. 

* **mean_PRF_dist_plot()**  
example (1D input): test 6  
example (2D input): test 7  
takes two time-series or 2D trajectory files. for each input, slices each into a 
number of windows. if inputs are time-series, embeds each window. gets PRF for each 
window. selects subset of window PRFs, computes their mean, plots distance to mean 
PRF vs time.

