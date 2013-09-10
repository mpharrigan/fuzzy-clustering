"""Do analysis on MSMs after they have been built."""

from __future__ import division
from matplotlib import pyplot as pp
import numpy as np
from msmbuilder import msm_analysis as msma


def get_implied_timescales(t_matrix, n_timescales=4, lag_time=1):
    """Get implied timescales from a transition matrix."""
    try:
        vals, vecs = msma.get_eigenvectors(t_matrix, n_eigs=n_timescales + 1)

        implied_timescales = -lag_time / np.log(vals[1:])
        implied_timescales_pad = np.pad(implied_timescales,
                                        (0, n_timescales - len(implied_timescales)),
                                        mode='constant')
        return implied_timescales_pad
    except Exception:
        print "+++ Error getting implied timescales +++"
        return np.zeros(n_timescales)

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


def _logistic(x, center=0.5, tension=100):
    return 1.0 / (1.0 + np.exp((-x + center) * tension))

def _expo(x, strength=5.0):
    return np.exp(strength * x) / np.exp(strength * 1.0)

def plot_points_with_alpha(points, memberships):
    """Plot points and their memberships.

    Points are colored based on the cluster in which it has the highest
    membership, and the alpha value is set to that membership value.
    """
    base_colors = [
                   [224, 27, 27],  # Red
                   [27, 224, 76],  # Green
                   [250, 246, 0],  # Yellow
                   [0, 29, 250],  # Blue
                   [221, 60, 230],  # Light Purple
                   [209, 140, 13],  # Brown
                   [5, 243, 255],  # Cyan
                   [102, 24, 204]  # Dark purple
                   ]
    base_colors = np.array(base_colors) / 255.


    assert len(memberships) == len(points), \
        'Membership (%d) and points (%d) must match' \
        % (len(memberships), len(points))
    n = len(memberships)

    colors = np.zeros((n, 4))
    for i in xrange(n):
        most_centroid = np.argmax(memberships[i])
        max_occupation = memberships[i, most_centroid]

        # Color based on what it belongs to most
        colors[i, 0:3] = base_colors[most_centroid % len(base_colors)]
        # Alpha based on degree
        colors[i, 3] = _expo(max_occupation)
        colors[i, 3] = 1.0


    pp.scatter(points[:, 0], points[:, 1], c=colors)
