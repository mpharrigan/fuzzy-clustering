.. fuzzy documentation master file, created by
   sphinx-quickstart on Thu Aug 29 10:55:14 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================
Fuzzy MSM Builder
=================================

This package provides tools to build and analyze a 
hidden markov model (HMM) with gaussian emission probabilities.

It also includes two IPython notebooks to demonstrate the effectiveness of this
scheme on the 2D Muller potential.

Using an HMM should allow us to use far fewer states in describing the kinetics.
It also lends itself better to using information criteria in determining the best
number of states. For example, the HMM code uses a gaussian mixture model with
maximal BIC.

It might allow us to use shorter lag times as well, while preserving markovity.
...to be tested...


.. include:: modules.rst
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

