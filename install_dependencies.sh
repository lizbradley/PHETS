# popt
wget http://rpm5.org/files/popt/popt-1.16.tar.gz
tar -xvzf popt-1.16.tar.gz
cd popt-1.16
./configure --prefix=/usr --disable-static &&
make
sudo make install
export LD_RUN_PATH="/usr/lib"

# ffmpeg
sudo apt-get install ffmpeg

# gnuplot
sudo apt-get install gnuplot

# for matplotlib
sudo apt-get install python-tk