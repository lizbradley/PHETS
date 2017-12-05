.. PHETS documentation master file, created by
   sphinx-quickstart on Sun Nov 19 00:57:42 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PHETS's documentation!
=================================

Introduction
------------


This package offers high-level tools for exploration and visualization of delay coordinate
embedding and persistent homology. It is used to investigate the utilization of these tools together as a signal
processing technique.

PHETS encompasses four submodules:

.. toctree::
   :maxdepth: 2

   signals
   PH
   DCE
   PRFstats



``signals`` holds the ``TimeSeries`` and ``Trajectory`` classes, which can be
initialized from arrays or text files. Calling the ``embed`` method of a
``TimeSeries`` returns a ``Trajectory``; calling the ``project`` method of
``Trajectory`` returns a ``TimeSeries``. ``TimeSeries`` and ``Trajectory`` both inherit from
``BaseTrajectory``, where all cropping, windowing, and normalization is handled.

``PH`` holds the ``Filtration`` class, which is initialized from a ``Trajectory`` and a
dict of filtration parameters. Filtration movies, persistence diagrams, and
persistence rank functions are created by calling the respective methods of
the Filtration class.

``DCE`` holds the embed function, as well as functions for generating embedding
movies. The movies functions take one or more ``TimeSeries`` and return one or
more ``Trajectory`` objects (created in the process of building the movies).

``PRFstats`` holds functions for statistical analysis of PRFs. Generally, they
take one or two ``Trajectory`` objects, create PRFs from the windows of the the
``Trajectory`` objects, do some analysis, and then save plots from the results.





macOS Installation Troubleshooting
==================================
PHETS requires the OpenMP C library `omp.h`. From what I can tell, OpenMP
is not included in clang (the default C compiler on macOS), and may only be
installed /configured for recent versions, and not with great ease. For these reasons, we've
never tried to run PHETS on clang, and cannot guarantee it will work
correctly.

On the other hand, OpenMP works with gcc out of the box, and you
may already have a version of gcc installed. If so, determine the version and edit
`find_landmarks_c_compile_str` in `config.py` to match. (NOTE: on macOS,
`gcc` is a symlink for clang. This is avoided by including the version number,
eg `gcc-5`.)

If you do not have gcc installed, you can do
```bash
brew install gcc
```
and then, as above, tweak `config.py`. You can also tell brew to install a
particular version if you would like (anything 5+ should work).

A quick way to test if things are working is to run ```python refresh.py``` in
the PHETS directory. This script will remove a number of temporary files and
attempt to compile `find_landmarks.c`

If the compiler is still giving errors (don't mind warnings), try
```bash
brew upgrade gcc
```
or
```bash
brew reinstall gcc --without-multilib
```

See [here](https://stackoverflow.com/questions/35134681/installing-openmp-on-mac-os-x-10-11)
and [here](https://stackoverflow.com/questions/29057437/compile-openmp-programs-with-gcc-compiler-on-os-x-yosemite)
for more information.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. * :ref:`modindex`
