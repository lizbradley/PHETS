.. PHETS documentation master file, created by
   sphinx-quickstart on Sun Nov 19 00:57:42 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the PHETS documentation!
===================================

Introduction
------------

This package offers high-level tools for exploration and visualization
of delay coordinate embedding and persistent homology. It is used to
investigate the utilization of these together as a signal processing
technique.

PHETS encompasses four submodules:

- :py:mod:`signals`
- :py:mod:`phomology`
- :py:mod:`embed`
- :py:mod:`prfstats`


``signals`` holds the ``TimeSeries`` and ``Trajectory`` classes, which can be
initialized from arrays or text files. Calling the ``embed`` method of a
``TimeSeries`` returns a ``Trajectory``; calling the ``project`` method of
``Trajectory`` returns a ``TimeSeries``. ``TimeSeries`` and ``Trajectory`` both
inherit from ``BaseTrajectory``, where all cropping, windowing, and
normalization is handled.

``phomology`` holds the ``Filtration`` class, which is initialized from a
``Trajectory`` and a dict of filtration parameters. Filtration movies,
persistence diagrams, and persistence rank functions are created by calling the
respective methods of the Filtration class.

``embed`` holds the ``embed`` function, as well as functions for generating
movies. The movies functions take one or more ``TimeSeries`` and return one or
more ``Trajectory`` objects (created in the process of building the movies).

``prfstats`` holds functions for statistical analysis of PRFs. Generally, they
take one or two ``Trajectory`` objects, create PRFs from the windows of the the
``Trajectory`` objects, do some analysis, and then save plots from the results.




Troubleshooting
---------------

compiling ``find_landmarks.c`` on OSX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PHETS requires the OpenMP C library ``omp.h``. From what I can tell, OpenMP
is not included in clang (the default C compiler on macOS), and may only be
installed /configured for recent versions, and not with great ease. For these reasons, we've
never tried to run PHETS on clang, and cannot guarantee it will work
correctly.

On the other hand, OpenMP works with gcc out of the box, and you
may already have a version of gcc installed. If so, determine the version and edit
``find_landmarks_c_compile_str`` in ``config.py`` to match. (NOTE: on macOS,
``gcc`` is a symlink for clang. This is avoided by including the version number,
eg ``gcc-5``.)

If you do not have gcc installed, you can do ``brew install gcc``
and then, as above, tweak ``config.py``. You can also tell brew to install a
particular version if you would like (anything 5+ should work).

A quick way to test if things are working is to run ``python refresh.py`` in
the PHETS directory. This script will remove a number of temporary files and
attempt to compile ``find_landmarks.c``

If the compiler is still giving errors (don't mind warnings), try
``brew upgrade gcc`` or ``brew reinstall gcc --without-multilib``

See `here <https://stackoverflow.com/questions/35134681/installing-openmp-on-mac-os-x-10-11>`_
and `here <https://stackoverflow.com/questions/29057437/compile-openmp-programs-with-gcc-compiler-on-os-x-yosemite>`_
for more information.


Regression Tests
---------------------
pytest is used for testing. To run the test suite, type ``pytest --tb=short``
from the top-level directory. (Running pytest within a subdirectory will only
execute the tests for that submodule.)

Each submodule contains a ``unit_test`` directory. The tests themselves are
defined in ``unit_tests/test__<submodule>.py``. These are not exactly
unit tests -- rather, each one calls or initializes a user-facing feature and
compares the result to a saved reference. The input data is found in
``unit_tests/data`` and the references in ``<unit_tests/ref>``.

``prfstats/unit_tests/prepare__data.py`` should be run regularly.
In the case of the ``prfstats`` module, in order to keep test execution time
lwo, the input data is pre-computed sets of Filtration instances. A small,
correct change to PHETS can break Python's ability to load these objects from
file, breaking the tests. In this case, run the routines in
``prfstats/unit_tests/prepare__data.py``, and the tests should work correctly.
Further, while an incorrect change should make at least one or two tests fail,
many ``prfstats`` tesss will not fail until the test data is re-generated with
``prepare__data.py``. If a couple tests fail, run this file to see what else is
broken down the pipeline, and always do so when a major change is completed.

Routines in ``unit_tests/prepare__refs.py`` should be run `only when you wish
to change the behavior of existing functionality`. They should also be run and
individually (that is, don't change the refs for features that you aren't
intentionally modifying).

To Do
----
Here are a couple ideas for how PHETS can be improved:

- speed up generation of movies by streaming images to ffmpeg over stdin rather than saving to file, `like so <https://stackoverflow.com/a/13298538>`_

- rewrite embed.embed with numpy for performance -- can it be done without a for loop?

- improve ``build_filtration.py`` readability

- implement switch for memory profiling of filtration construction

- integrate new ``utilities.timeit`` decorator

- find a faster way than text files to pass data from ``find_landmarks`` to Python

- streamline passage of params to ``find_landmarks``


Documentation
------------------
This documentation is built with Sphinx. The autodoc extension is used to
generate the `library reference <Reference>`_ from docstrings in the Python
code. The text and layout for all other sections (eg this paragraph) is defined
in ``docs/source/index.rst``.

To build this documentation, TeX must be installed, along with the following:

- texlive-latex-recommended
- texlive-fonts-recommended
- texlive-latex-extra
- latexmk

I used ``sudo apt-get install <package>`` for each.

To update the documentation:

.. code-block:: bash

   cd docs
   make latexpdf
   mv build/latex/PHETS.pdf ../documentation.pdf



Reference
---------
.. toctree::
   signals
   phomology
   embed
   prfstats

