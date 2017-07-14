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

See reference.pdf and *Tester.py files.

