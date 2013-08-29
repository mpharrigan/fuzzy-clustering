from __future__ import division
from fuzzy import mixture, get_data
from fuzzy.old import euclidean, fcm
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
    """Plot multiple trajectories in different colors."""
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

def plot_lambda_bar(implied_timescales, descs):

    little_sep = 1.0
    width = little_sep
    big_sep = little_sep * (len(implied_timescales) + 2)
    max_n_eigen = 0

    colors = ['r', 'b', 'y', 'g', 'k']

    pp.clf()
    for i in xrange(len(implied_timescales)):
        ool = implied_timescales[i]
        xlocs = np.arange(0, len(ool) * big_sep, big_sep) + i * little_sep
        pp.bar(xlocs, ool, width=width, color=colors[i], label=descs[i])
        if len(ool) > max_n_eigen:
            max_n_eigen = len(ool)

    avg_offset = len(implied_timescales) * little_sep / 2.0
    xlocs = np.arange(0, max_n_eigen * big_sep, big_sep) + avg_offset
    pp.xticks(xlocs, ["Eigenvalue %d" % (i + 1) for i in range(max_n_eigen)])
    pp.legend()

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
    """From an hmm, generate a list of trajectories."""
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

def compare_t_matrices(t_matrix1, t_matrix2, desc="T-matrix compare"):
    pp.clf()
    plot_t_matrix(t_matrix1, 'b')
    plot_t_matrix(t_matrix2, 'r')
    pp.title(desc)

def test_hmm(n_trajs=10, traj_len=1000, show=False):
    """Test various schemes."""

    # Build model and sample
    hmm = build_sample_hmm()
    traj_list = sample_from_hmm(hmm, n_trajs, traj_len)
    constructed_t_matrix = _get_t_matrix(hmm)
    constructed_lambda = fcm.analyze_msm(constructed_t_matrix, None, "Construction", neigen=3, show=False)

    # View trajectories
    plot_trajectories(traj_list)
    if show: pp.show()

    # Do HMM
    hmm_t_matrix = mixture.hmm(traj_list, min_k=2, max_k=10, lag_time=1)
    hmm_lambda = fcm.analyze_msm(hmm_t_matrix, None, "HMM", neigen=3, show=False)


    # Compare matrices
    compare_t_matrices(constructed_t_matrix, hmm_t_matrix, "Constructed vs. HMM")
    if show: pp.show()

    # Do classic
    clustering.logger.setLevel('ERROR')
    metric = euclidean.Euclidean2d()
    shim_t = get_data.get_shimtraj_from_trajlist(traj_list)
    classic_t_matrix = fcm.classic(shim_t, n_clusters=3, n_medoid_iters=10, metric=metric, lag_time=1, show=False)
    classic_lambda = fcm.analyze_msm(classic_t_matrix, None, desc="Classic", neigen=3, show=False)

    # Compare matrices
    compare_t_matrices(constructed_t_matrix, classic_t_matrix, "Constructed vs. Classic")
    if show: pp.show()

    # Do big k classic
    big_classic_t_matrix = fcm.classic(shim_t, n_clusters=200, n_medoid_iters=10, metric=metric, lag_time=1, show=False)
    big_classic_lambda = fcm.analyze_msm(big_classic_t_matrix, None, desc="Classic (large k)", neigen=5, show=False)

    # Plot Eigenvalues
    plot_lambda_bar([constructed_lambda, hmm_lambda, classic_lambda, big_classic_lambda], ('Construction', 'HMM', 'Classic k=3', 'Classic k=200'))
    pp.show()

    return constructed_t_matrix, hmm_t_matrix, classic_t_matrix


if __name__ == "__main__":
    test_hmm()
