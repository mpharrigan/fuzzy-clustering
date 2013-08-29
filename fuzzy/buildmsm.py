"""Code for building MSMs in some sort of fuzzy scheme."""

from __future__ import division
from msmbuilder import MSMLib as msm
import numpy as np
import scipy.sparse


def get_counts_from_traj_soft1(states, n_states=None, lag_time=1):
    if not n_states:
        n_states = np.max(states) + 1


    from_states = states[:-lag_time: 1]
    to_states = states[lag_time:: 1]


    assert len(from_states) == len(to_states)

    soft_counts = np.zeros((n_states, n_states))
    for i in xrange(len(from_states)):
        soft_counts += np.outer(from_states[i], to_states[i])

    sparse_counts = scipy.sparse.coo_matrix(soft_counts)
    return sparse_counts

def _corr_product(from_points, to_points):
    """Perform the correlation product (which is just a normalized dot product
    according to eq (5) in Tavan 2005.
    """
    assert len(from_points) == len(to_points), 'number of points must match'

    result = np.dot(from_points, to_points)
    return result

def _crr(r, rp, from_states, to_states):
    """Find transfer matrix element r, r' from lists of from states and
    to states

    This is according to eq (6) in Tavan 2005.
    """
    from_points_rp = from_states[:][rp]
    to_points_r = to_states[:][r]

    numerator = _corr_product(to_points_r, from_points_rp)
    denominat = _corr_product(from_points_rp, from_points_rp)
    denominat = np.sqrt(denominat)
    result = numerator / denominat
    return result

def _crr2(r, rp, time_pairs):
    """Find transfer matrix element r, r' from a matrix of time pairs

    This is according to eq (6) in Tavan 2005.
    """
    from_points_rp = time_pairs[:, 0, rp]
    to_points_r = time_pairs[:, 1, r]

    numerator = _corr_product(to_points_r, from_points_rp)
    denominat = _corr_product(from_points_rp, from_points_rp)
    denominat = np.sqrt(denominat)
    result = numerator / denominat
    return result

def get_counts_from_traj_soft2(states, n_states=None, lag_time=1):
    """Try to get a soft count matrix according to Tavan 2005.

    states is a chronological list of states from which this function
    will calculate time pairs.
    """
    if not n_states:
        n_states = np.max(states) + 1


    from_states = states[:-lag_time: 1]
    to_states = states[lag_time:: 1]

    soft_counts = np.zeros((n_states, n_states))
    for i in xrange(n_states):
        for j in xrange(n_states):
            soft_counts[i, j] = _crr(i, j, from_states, to_states)

#     import pdb; pdb.set_trace()
    sparse_counts = scipy.sparse.coo_matrix(soft_counts)
    return sparse_counts

def get_counts_from_pairs(time_pairs, n_states):
    """Get a soft count matrix by using the outer product

    time_pairs is a (n_points, 2, n_states) shaped matrix containing pairs
    of membership vectors.
    """
    soft_counts = np.zeros((n_states, n_states))
    n_pairs = len(time_pairs)

    for pair_i in xrange(n_pairs):
        # For each pair, do the outer product between from and to membership
        # vectors and add it to the building count matrix
        soft_counts += np.outer(time_pairs[pair_i, 0, :], time_pairs[pair_i, 1, :])

    sparse_counts = scipy.sparse.coo_matrix(soft_counts)
    return sparse_counts

def build_from_memberships(memberships, lag_time=1):
    """Build an MSM from a time array of membership vectors."""

    # Sliding window
    from_states = memberships[:-lag_time: 1]
    to_states = memberships[lag_time:: 1]

    assert len(from_states) == len(to_states)

    n_pairs = len(from_states)
    n_times = 2
    n_clusters = memberships.shape[1]

    pairs = np.zeros((n_pairs, n_times, n_clusters))
    pairs[:, 0, :] = from_states
    pairs[:, 1, :] = to_states

    counts = get_counts_from_pairs(pairs, n_clusters)
    rev_counts, t_matrix, populations, mapping = msm.build_msm(counts)
    return rev_counts, t_matrix, populations, mapping

def build_classic_from_memberships(memberships, lag_time=1):
    """Build a classic msm by turning a membership array into a state list.

    This function uses msmbuilder code to calculate the count matrix. Use this
    for compairing quantized versions of the fuzzy count matrix building
    for consistency.
    """
    states = np.zeros(memberships.shape[0], dtype='int')
    n_states = memberships.shape[1]

    for i in xrange(memberships.shape[0]):
        memb = memberships[i]
        state = np.argmax(memb)
        states[i] = state

    counts = msm.get_counts_from_traj(states, n_states, lag_time)
    rev_counts, t_matrix, populations, mapping = msm.build_msm(counts)
    return rev_counts, t_matrix, populations, mapping


