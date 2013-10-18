Hidden Markov Models to Desribe Protein Dynamics
==================

Motivation
--------------
Traditionally, we have clustered conformation-space into many discrete
voronoi cells and used this clustering to build a large transition matrix.
This results in implied timescales that are consistently faster than they
should be. If a conformmation diffuses on the boundary between two states, it generates many counts, leading to too fast timescales.


![two-well](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/two-well.png)


The transition matrix is supposed to estimate the true propagator of the system. In a hard state MSM, the eigenfunctions are constrained to be step functions. It would make sense to use a basis set that matches the physics of the system.

Results on a toy system
----------

### IT vs number of clusters


Closer to solid black line is better

![it-vs-k](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/its_vs_k.png)

### IT vs lag time


Closer to solid black line is better

![it-vs-lt](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/its_vs_lt.png)

### Eigenvectors


In a traditional MSM, the eigenvectors are constrained to be blocky voronoi cells of solid color (not pictured). An HMM gives a pretty good description of the true eigenvectors.

![eigenvectors](https://raw.github.com/mpharrigan/fuzzy-clustering/master/figs/eigens.png)