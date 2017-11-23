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
dict of filtration parameters. Filtration moves, persistence diagrams, and
persistence rank functions are created by calling the respective methods of
the Filtration class.

``DCE`` holds the embed function, as well as functions for generating embedding
movies. The movies functions take one or more ``TimeSeries`` and return one or
more ``Trajectory`` objects (created in the process of building the movies).

``PRFstats`` holds functions for statistical analysis of PRFs. Generally, they
take one or two ``Trajectory`` objects, create PRFs from the windows of the the
``Trajectory`` objects, do some analysis, and then save plots from the results.



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. * :ref:`modindex`
