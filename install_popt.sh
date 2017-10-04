wget http://rpm5.org/files/popt/popt-1.16.tar.gz
tar -xvzf popt-1.16.tar.gz
cd popt-1.16
./configure --prefix=/usr --disable-static &&
make
sudo make install
export LD_RUN_PATH="/usr/lib"
rm -r -f popt-1.16
rm popt-1.16.tar.gz