from __future__ import division
from fuzzy import mixture, get_data, euclidean, fcm
from matplotlib import pyplot as pp
import ghmm
import numpy as np
import scipy.sparse
from msmbuilder import clustering

def _get_t_matrix(hmm):
    """Get the transition matrix in an msmbuilder format from an HMM.

    This is a scipy.sparse.csr_matrix.
    """
    t_matrix = hmm.asMatrices()[0]
    return scipy.sparse.csr_matrix(t_matrix)


def _row_normalize(t_matrix):
    """Row-normalize an unnormalized transition matrix."""
    i = 0
    for i in xrange(len(t_matrix)):
        row = t_matrix[i]
        norm = sum(row)
        t_matrix[i] = [r / norm for r in row]
    return t_matrix



def plot_trajectories(traj_list):
    pp.clf()
    for traj in traj_list:
        pp.plot(traj[:, 0], traj[:, 1], 'o-')

def plot_t_matrix(t_matrix, color, alpha=0.5):
    assert t_matrix.shape[0] == t_matrix.shape[1]
    n_states = t_matrix.shape[0]

    for column in xrange(n_states):
        for row in xrange(n_states):
            size = 10 * np.exp(t_matrix[row, column])
            size = 50 * np.sqrt(t_matrix[row, column])
            pp.plot(column, n_states - row - 1, '%so' % color, markersize=size, alpha=alpha)

    pp.xlim(-1, n_states)
    pp.ylim(-1, n_states)



def build_sample_hmm():
    """Build an HMM from a constructed transition matrix and artificial
    emission probabilities.
    """

    # Set up system
    t_matrix = [
                [10, 2, 5],
                [1, 10, 0],
                [5, 1, 10]]
    t_matrix = _row_normalize(t_matrix)

    means = [
             [0.0, 0.0],
             [6.0, -1.1],
             [6.8, 6.8]]

    covars = [0] * len(means)
    covars[0] = [[0.5, 0.0],
                 [0.0, 0.5]]
    covars[1] = [[0.5, 1.0],
                 [1.0, 0.5]]
    covars[2] = [[0.3, 0.3],
                 [0.0, 0.5]]
    covars = np.array(covars)

    initial_occupancy = [1.0 / len(means)] * len(means)
    print(initial_occupancy)

    # matrix arrangement
    emissions = [[means[j], covars[j].flatten()] for j in xrange(len(means))]

    g_float = ghmm.Float()
    g_distribution = ghmm.MultivariateGaussianDistribution(g_float)

    hmm = ghmm.HMMFromMatrices(g_float, g_distribution, t_matrix,
                             emissions, initial_occupancy)

    return hmm



def sample_from_hmm(hmm, n_trajs, traj_len):
    domain = hmm.emissionDomain
    dim = hmm.cmodel.dim

    # Sample
    seq_set = hmm.sample(n_trajs, traj_len)
    seq_list = [map(domain.externalSequence, seq) for seq in seq_set]

    # Unflatten
    traj_list = list()
    for seq in seq_list:
        traj = np.array(seq)
        traj = np.reshape(traj, (len(seq) / dim, dim))
        traj_list.append(traj)

    return traj_list



def test_hmm(n_trajs=5, traj_len=200):
    hmm = build_sample_hmm()
    traj_list = sample_from_hmm(hmm, n_trajs, traj_len)
    plot_trajectories(traj_list)
    if False: pp.show()

    t_matrix = mixture.hmm(traj_list, min_k=2, max_k=10, lag_time=1, n_eigen=4, show=False)
    constructed_t_matrix = _get_t_matrix(hmm)

    pp.clf()
    plot_t_matrix(t_matrix, 'b')
    plot_t_matrix(t_matrix, 'r')
    if True: pp.show()

    # Do classic
    clustering.logger.setLevel('ERROR')
    metric = euclidean.Euclidean2d()
    shim_t = get_data.get_shimtraj_from_trajlist(traj_list)
    classic_t_matrix = fcm.classic(shim_t, n_clusters=3, n_medoid_iters=10, metric=metric, lag_time=1, show=False)


    pp.clf()
    plot_t_matrix(constructed_t_matrix, 'b')
    plot_t_matrix(classic_t_matrix, 'r')
    pp.show()


    # Do classic again with many states
    shim_t = get_data.get_shimtraj_from_trajlist(traj_list)
    classic_t_matrix = fcm.classic(shim_t, n_clusters=100, n_medoid_iters=10, metric=metric, lag_time=1, show=False)


    pp.clf()
    plot_t_matrix(constructed_t_matrix, 'b')
    plot_t_matrix(classic_t_matrix, 'r')
    pp.show()

    return t_matrix, constructed_t_matrix, classic_t_matrix


if __name__ == "__main__":
    test_hmm()
