Hidden Markov Models to Desribe Protein Dynamics
==================

Motivation
--------------
Traditionally, we have clustered conformation-space into many discrete
voronoi cells and used this clustering to build a large transition matrix.
This results in implied timescales that are consistently faster than they
should be. 

The motivation of this work can be found in considering a simple, two-well,
1D potential. What is the optimal clustering? There are only two physically
stable states, but if we used two hard states, conformations on the cusp
of the barrier are considered to be the same as conformations at the bottom of the well.

![two-well](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/two-well.png)

If the particle moving under this potential diffuses for a bit at the top of the potential, crossing the line that distinguishes our two hard states, it generates many counts, distorting the dynamics. 

The theoretical underpinnings of Markov State Models is that they are used to estimate the propagator. The typical approach is to use indicator basis functions to approximate the propagator. This gives step-like
eigenfunctions whose resolution increases with increasing number of states: above figure, red line.  One can imagine using a different basis set instead. Two Gaussian functions, for example, could describe
dynamics on this potential much better than even a high number of step functions with many fewer parameters.

IT vs number of clusters
------------

Closer to solid black line is better

![it-vs-k](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/its_vs_k.png)

IT vs lag time
---------------

Closer to solid black line is better

![it-vs-lt](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/its_vs_lt.png)

Eigenvectors
----------------

In a traditional MSM, the eigenvectors are constrained to be blocky voronoi cells of solid color (not pictured)

![eigenvectors](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/eigens.png)