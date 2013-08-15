from __future__ import division
from msmbuilder import MSMLib as msm
import numpy as np
import scipy.sparse

eps = 1.0e-10

def get_counts_from_traj(states, n_states=None, lag_time=1, sliding_window=True):
    """Computes the transition count matrix for a sequence of states (single trajectory).

    Parameters
    ----------
    states : array
        A one-dimensional array of integers representing the sequence of states.
        These integers must be in the range [0, n_states]
    n_states : int
        The total number of states. If not specified, the largest integer in the
        states array plus one will be used.
    lag_time : int, optional
        The time delay over which transitions are counted
    sliding_window : bool, optional
        Use sliding window

    Returns
    -------
    C : sparse matrix of integers
        The computed transition count matrix
    """

    msm.check_assignment_array_input(states, ndim=1)

    if not n_states:
        n_states = np.max(states) + 1

    if sliding_window:
        from_states = states[:-lag_time: 1]
        to_states = states[lag_time:: 1]
    else:
        from_states = states[:-lag_time: lag_time]
        to_states = states[lag_time:: lag_time]
    assert from_states.shape == to_states.shape
    
    # import pdb; pdb.set_trace()

    transitions = np.row_stack((from_states, to_states))
    counts = np.ones(transitions.shape[1], dtype=int)
    try:
        C = scipy.sparse.coo_matrix((counts, transitions),
                                    shape=(n_states, n_states))
    except ValueError:
        # Lutz: if we arrive here, there was probably a state with index -1
        # we try to fix it by ignoring transitions in and out of those states
        # (we set both the count and the indices for those transitions to 0)
        mask = transitions < 0
        counts[mask[0, :] | mask[1, :]] = 0
        transitions[mask] = 0
        C = scipy.sparse.coo_matrix((counts, transitions),
                                    shape=(n_states, n_states))

    return C


def outernorm(v1, v2, which='columns'):
    mat = np.outer(v1, v2)
    
    if which == 'columns':
        axis = 0
    elif which == 'rows':
        axis = 1
    else:
        raise ValueError()
    
    sums = np.sum(mat, axis=axis)
    for i in xrange(mat.shape[axis]):
        if sums[i] > eps:
            if which == 'columns':
                mat[:, i] = mat[:, i] / sums[i]
            elif which == 'rows':
                mat[i] = mat[i] / sums[i]
                
    return mat
    

def get_counts_from_traj_soft1(states, n_states=None, lag_time=1):

    # check_assignment_array_input(states, ndim=1)

    if not n_states:
        n_states = np.max(states) + 1

#     if sliding_window:
    from_states = states[:-lag_time: 1]
    to_states = states[lag_time:: 1]

#     else:
#         from_states = states[: -lag_time: lag_time]
#         to_states = states[lag_time:: lag_time]
    assert len(from_states) == len(to_states)
    
    soft_counts = np.zeros((n_states, n_states))
    for i in xrange(len(from_states)):
        soft_counts += np.outer(from_states[i], to_states[i])

    C = scipy.sparse.coo_matrix(soft_counts)
    return C

def _corr_product(from_points, to_points):
    """Perform the correlation product (which is just a normalized dot product
    according to eq (5) in Tavan 2005.
    """    
    assert len(from_points) == len(to_points), 'number of points must match'
    n_points = len(from_points)
    
    result = np.dot(from_points, to_points) / (n_points - 1)
    # TODO: Do we divide by number of pairs?
    result = np.dot(from_points, to_points)
    return result

def _crr(r, rp, from_states, to_states):
    """Find transfer matrix element r, r'
    
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
    """Find transfer matrix element r, r'
    
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
    """Try to get a soft count matrix according to Tavan 2005"""
    if not n_states:
        n_states = np.max(states) + 1

    
    from_states = states[:-lag_time: 1]
    to_states = states[lag_time:: 1]
    
    soft_counts = np.zeros((n_states, n_states))
    for i in xrange(n_states):
        for j in xrange(n_states):
            soft_counts[i, j] = _crr(i, j, from_states, to_states)
        
#     import pdb; pdb.set_trace()
    C = scipy.sparse.coo_matrix(soft_counts)
    return C      

def get_counts_from_pairs(time_pairs, n_states):
    soft_counts = np.zeros((n_states, n_states))
    for i in xrange(n_states):
        for j in xrange(n_states):
            soft_counts[i, j] = _crr2(i, j, time_pairs)
        
#     import pdb; pdb.set_trace()
    C = scipy.sparse.coo_matrix(soft_counts)
    return C   
            
    
def get_counts_from_traj_soft(states, n_states=None, lag_time=1):
    return get_counts_from_traj_soft2(states, n_states, lag_time)
