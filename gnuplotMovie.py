#! /usr/bin/env python
# $Id: demo.py 299 2007-03-30 12:52:17Z mhagger $

# Copyright (C) 1999-2003 Michael Haggerty <mhagger@alum.mit.edu>
#
# This file is licensed under the GNU Lesser General Public License
# (LGPL).  See LICENSE.txt for details.

"""demo.py -- Demonstrate the Gnuplot python module.

Run this demo by typing 'python demo.py'.  For a more complete test of
the Gnuplot package, see test.py.

"""

from numpy import *

# If the package has been installed correctly, this should work:
import Gnuplot, Gnuplot.funcutils


def demo():
    """Demonstrate the Gnuplot package."""

    # A straightforward use of gnuplot.  The `debug=1' switch is used
    # in these examples so that the commands that are sent to gnuplot
    # are also output on stderr.
    g = Gnuplot.Gnuplot()
    g.title('A simple example') # (optional)

    # g('set data style linespoints') # give gnuplot an arbitrary command

    # g('set object 1 polygon from 0.4, 0.7, to 0.0, 0,0, to 0.9, 0.01')
    # g("set object 1 fc rgb '#999999' fillstyle solid lw 0")
    # g.plot([[0,1.1], [1,5.8], [2,3.3], [3,4.2]])




    g('set style fill transparent solid 0.6')




    d = Gnuplot.Data([[0, 0], [1, 0], [1, 1]],
                     title='sdfs',
                     with_='filledcurves closed lw 2')


    g.plot(d)
    g.hardcopy('gp_test.ps', enhanced=1, color=1)


    g.reset()

if __name__ == '__main__':
    demo()
