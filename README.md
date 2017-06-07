## Synopsis

**P**ersistent **H**omology on **E**mbedded **T**ime-**S**eries

This repository offers high-level tools for exploration and visualization of
* delay coordinate embedding from 1D to 2D
* persistent homology in 2D

and is used to investigate the application of these procedures as a signal processing technique.

Also included is a dataset of time-series (mostly musical instrument recordings) and higher dimensional trajectories in .txt.
3D functionality is under construction.

## Installation

### Dependencies
* Python 2.7
* GCC 5
* popt.h
* ffmpeg (in PATH variable)


### Setup
```bash
git clone https://github.com/eeshugerman/PHETS.git
cd PHETS
pip install -r requirements.txt
```



## Usage/Reference

Presently, functionality is split into three modules. Each module offers a handful of related functions and is driven by 
its own tester file. Each tester contains many tests which have accumulated over time. Often, older tests are not kept up
to date and will no longer run, but are kept for record keeping purposes.

Tests may be run by either setting the test variable at the top of the tester script, or by uncommenting the following line
and providing the test number as a command line argument.


#### DCE: delay coordinate embedding

DCE provides three main functions which each generate a different type of movie:

* vary_tau()
..* example: test 4
..* takes one input file shows embedding over range of tau

* slide_window()
..* takes one input file, shows embedding over a range of windows
..* example: test 5

* compare_vary_tau()
..* like vary_tau(), but takes two input files and shows their embeddings side by side.
..*

* compare_multi()
..* takes two directories of (eg one with range of piano notes, another with range of viol notes), and generates a movie
over a range note indexes (pitch). Tau and crop can be set explicity or automatically.


#### PH: persistent homology

* make_movie()
..* make filtration movie
* make_PD()
..* make persistent diagram
* make_PRF_plot()
..* make persistent rank function plot

#### PRFCompare: persistent rank function comparision


* PRF_dist_plot()
..* takes range of time-series files and reference file (by index). Generates PRF for each, and finds distance to reference PRF,
plots distance vs index. 

* mean_PRF_dist_plot()
..* take two time-series or 2D trajectory files. For each input, slices each into a number of windows. If inputs are time-series, embeds
each window. Finds PRF for each window. Takes subset of window PRFs for each input, computes their mean, plots distance to mean PRF
vs time.

