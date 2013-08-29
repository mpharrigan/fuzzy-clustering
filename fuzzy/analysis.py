"""Do analysis on MSMs after they have been built."""

from __future__ import division
from matplotlib import pyplot as pp
import numpy as np
from msmbuilder import msm_analysis as msma


def get_implied_timescales(t_matrix, n_timescales=4, lag_time=1):
    """Get implied timescales from a transition matrix."""
    vals, vecs = msma.get_eigenvectors(t_matrix, n_eigs=n_timescales + 1)

    implied_timescales = -lag_time / np.log(vals[1:])
    implied_timescales_pad = np.pad(implied_timescales,
                                    (0, n_timescales - len(implied_timescales)),
                                    mode='constant')
    return implied_timescales_pad

def plot_lambda_bar(implied_timescales, descs, logplot=False):
    """Plot implied timescales on a bar chart."""

    little_sep = 1.0
    width = little_sep
    big_sep = little_sep * (len(implied_timescales) + 2)
    max_n_eigen = 0

    colors = ['b', 'r', 'y', 'g', 'k']

    pp.clf()
    for i in xrange(len(implied_timescales)):
        it = implied_timescales[i]
        xlocs = np.arange(0, len(it) * big_sep, big_sep) + i * little_sep
        pp.bar(xlocs, it, width=width, color=colors[i], label=descs[i], bottom=0.0, log=logplot)
        if len(it) > max_n_eigen:
            max_n_eigen = len(it)

    avg_offset = len(implied_timescales) * little_sep / 2.0
    xlocs = np.arange(0, max_n_eigen * big_sep, big_sep) + avg_offset
    pp.xticks(xlocs, ["Timescale %d" % (i + 1) for i in range(max_n_eigen)])
    pp.legend()
