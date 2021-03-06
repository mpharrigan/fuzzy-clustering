"""Use mixture modeling and HMM to build a transition matrix."""

from fuzzy import buildmsm, get_data
from matplotlib import pyplot as pp
from sklearn import mixture
import ghmm
import ghmmhelper
import ghmmwrapper
import numpy as np
import scipy.sparse.csr
from msmbuilder import msm_analysis as msma

def plot_distribution(mixture_model, grid, t_matrix=None, eigen=1, n_contours=80):
    """Plot the mixture distribution."""

    xx, yy = grid

    mixture_samples = np.c_[xx.ravel(), yy.ravel()]
    contour_data = mixture_model.score(mixture_samples)
    contour_data = -contour_data.reshape(xx.shape)

    if t_matrix is not None:
        _, vecs = msma.get_eigenvectors(t_matrix, n_eigs=eigen)
        sizes = vecs[:, -1] * 300
        print vecs[:, -1]
        colors = ['r' if s > 0 else 'b' for s in sizes]
        sizes = np.abs(sizes)
    else:
        sizes = 300
        colors = 'y'


    # Plot means
    means = mixture_model.means_
    pp.scatter(means[:, 0], means[:, 1], c=colors, s=sizes)

    pp.contour(xx, yy, contour_data, n_contours)

def print_ic(mixture_model, points, desc=''):
    """Status for finding a mixture model."""
    print("%s\tfit results: AIC: %g\tBIC: %g" %
          (desc, mixture_model.aic(points), mixture_model.bic(points)))


def quantize_memberships(memberships):
    """For testing purposes, quantize membership vectors to be step functions

    The new memberships will be of the form 0, 0, ..., 0, 1, 0, ..., 0. The
    occupied state will be the state with the highest occupation.

    Use this function to test that building an msm from memberships produces
    the same results as building an msm from a state list with the msmbuilder
    package.
    """
    n_clusters = memberships.shape[1]
    new_memberships = np.zeros_like(memberships)
    for i in xrange(len(memberships)):
        memb = memberships[i]
        state = np.argmax(memb)
        new_memb = np.zeros(n_clusters)
        new_memb[state] = 1.0
        new_memberships[i] = new_memb
    return new_memberships


def get_mixture_model(points, min_k, max_k, fix_k=None, mm_stride=1):
    """Find the best mixture model based on BIC."""
    prev_mm = None
    prev_bic = None
    min_mm = None

    if fix_k is None:
        for k in xrange(min_k, max_k):
            mm = mixture.GMM(n_components=k, covariance_type='full')
            mm.fit(points[::mm_stride])

            bic = mm.bic(points)
            print("Trying k = %d, BIC = %g" % (k, bic))

            if prev_bic is not None and bic > prev_bic:
                min_mm = prev_mm
                print("\tUsing k = %d" % (k - 1))
                break

            prev_bic = bic
            prev_mm = mm
            min_mm = mm
        del mm
    else:
        min_mm = mixture.GMM(n_components=fix_k, covariance_type='full')
        min_mm.fit(points[::mm_stride])

    return min_mm

def redo_mixture_model(optimized_hmm):
    matrices = optimized_hmm.asMatrices()
    means = np.array([matrices[1][j][0] for j in xrange(len(matrices[1]))])
    n_components = means.shape[0]
    n_features = means.shape[1]
    covars = np.array([matrices[1][j][1] for j in xrange(len(matrices[1]))])
    covars = np.reshape(covars, (n_components, n_features, n_features))
    # weights = np.array(matrices[2])
    weights = np.ones(len(matrices[2]))

    mm = mixture.GMM(n_components=n_components, params='', init_params='', covariance_type='full')
    mm.means_ = means
    mm.covars_ = covars
    mm.weights_ = weights

    return mm

def get_hidden_markov_model(mixture_model, guess_t_matrix):
    """Get an (unoptomized) hidden markov model from the mixture model and
    a guess at the transition matrix.

    The guess transition matrix is typically created by summing over the
    outer product of time-pairs of membership vectors.
    """

    # Emission  probabilities for HMM, using their very silly
    # matrix arrangement
    emissions = [[mixture_model.means_[j], mixture_model.covars_[j].flatten()]
                 for j in xrange(mixture_model.n_components)]

    # Initial transition matrix
    if isinstance(guess_t_matrix, scipy.sparse.csr.csr_matrix):
        guess_t_matrix = guess_t_matrix.todense()
        guess_t_matrix = guess_t_matrix.tolist()

    # Initial occupancy
    # Todo: figure out if initial occupancy matters
    initial_occupancy = ([1.0 / mixture_model.n_components]
                         * mixture_model.n_components)

    # Set up distribution
    g_float = ghmm.Float()
    g_distribution = ghmm.MultivariateGaussianDistribution(g_float)

    # Put it all together
    model = ghmm.HMMFromMatrices(g_float, g_distribution, guess_t_matrix,
                                 emissions, initial_occupancy)
    return model

def perform_optimization(hidden_mm, trajs, lag_time, sliding_window=True):
    """Optimize a hidden markov model given a list of trajectories.

    Use the Baum-Welch algorithm for learning the transition matrix, fixing
    emission probabilities.
    """

    # Domains for our multivariate gaussians
    domain = hidden_mm.emissionDomain

    # Do sliding window
    if sliding_window:
        # A naive way of doing this is by making many trajectories
        slides = xrange(lag_time)
        lagged_trajs = list()
        for i in xrange(len(trajs)):
            traj = trajs[i]
            for slide in slides:
                lagged_trajs.append(traj[slide::lag_time])
    else:
        lagged_trajs = [t[::lag_time] for t in trajs]

    # Prepare the trajectories by flattening them to 1D
    prepared_trajs = [t.flatten().tolist() for t in lagged_trajs]
    # Build the c-style sequences object manually
    (seq_c, lengths) = ghmmhelper.list2double_matrix(prepared_trajs)
    lengths_c = ghmmwrapper.list2int_array(lengths)
    cseq = ghmmwrapper.ghmm_cseq(seq_c, lengths_c, len(trajs))

    # Make a SequenceSet wrapper around the c-style object
    train_seq = ghmm.SequenceSet(domain, cseq)
    # Perform the Baum Welch optimization
    likelihood = hidden_mm.baumWelch(train_seq, nrSteps=10000000, loglikelihoodCutoff=1.0e-5)

    print "Final baum welch likelihood: {}".format(likelihood)

    return likelihood, hidden_mm


def hmm(traj_list, min_k=3, max_k=20, fix_k=None, lag_time=1,
        sliding_window=True, mm_stride=1):
    """Build a hidden markov model from a list of trajectories.

    This function will first create a mixture model and then use the
    Baum-Welch algorithm to learn the hidden transition matrix.

    The number of mixture components can be determined by maximizing
    the BIC or by specifying a fixed number

    :param traj_list: List of tranjectories
    :param min_k: The number at which to start searching for the optimal
                    number of mixture components
    :param max_k: The maximum number of mixture components to use.
    :param fix_k: If none, find the best number of mixture components within
                    restraints. Otherwise, use precisely fix_k number of
                    components
    :param lag_time: The lag time of the model. #TODO: This doesn't do
                    anything maybe
    :param sliding_window: Whether to use a sliding window with lag_time
    :type sliding_window: bool
    :returns: scipy.sparse.csr_matrix -- transition matrix
    :type min_k: int
    :type max_k: int
    :type fix_k: int or None
    :type traj_list: list
    :type lag_time: int
    """
    if not sliding_window:
        lt_stride = lag_time
    else:
        lt_stride = 1

    points = get_data.get_points_from_trajlist(traj_list)
    first_mixture_model = get_mixture_model(points, min_k, max_k, fix_k, mm_stride)
    memberships = first_mixture_model.predict_proba(points)

    # Build an initial MSM as a guess
    print("Building guess transition matrix from discretization of mixture model")
    rev_counts, t_matrix, populations, mapping = \
                buildmsm.build_classic_from_memberships(memberships, lag_time=lag_time)

    # Learn from trajectories in HMM
    print("Performing Baum-Welch algorithm")
    hidden_mm = get_hidden_markov_model(first_mixture_model, t_matrix)
    likelihood, hidden_mm = perform_optimization(hidden_mm, traj_list,
                                                 lag_time=lag_time,
                                                 sliding_window=sliding_window)

    opt_mixture_model = redo_mixture_model(hidden_mm)

    # Get the transition matrix in the normal form
    new_t_matrix = hidden_mm.asMatrices()[0]
    new_t_matrix = scipy.sparse.csr_matrix(new_t_matrix)

    return t_matrix, new_t_matrix, hidden_mm, first_mixture_model, opt_mixture_model, likelihood

# def test_mixture(min_k=3, max_k=20, fix_k=None, n_eigen=4, lag_time=10):
#     """Run a bunch of functions to test the various schemes."""
#
#     points = get_data.get_points(stride=1)
#     first_mixture_model = get_mixture_model(points, min_k, max_k, fix_k)
#
#     centroids = first_mixture_model.means_
#     fcm.plot_centroids(centroids)
#     plot_distribution(first_mixture_model)
#     # pp.show()
#
#     # Get the memberships
#     memberships = first_mixture_model.predict_proba(points)
#
#     # Pick highest membership, put it in a classic state transition list,
#     # and use straight msmbuilder code to build the count matrix
#     rev_counts, t_matrix, populations, mapping = \
#                     buildmsm.build_classic_from_memberships(memberships,
#                                                             lag_time=lag_time)
#     fcm.analyze_msm(t_matrix, centroids, desc='Old, Hard',
#                     neigen=n_eigen, show=False)
#
#     # Quantize the membership vectors and use the new method of building the
#     # count matrix and test with above
#     q_memberships = quantize_memberships(memberships)
#     rev_counts, t_matrix, populations, mapping = \
#                     buildmsm.build_from_memberships(q_memberships,
#                                                     lag_time=lag_time)
#     fcm.analyze_msm(t_matrix, centroids, "New, Hard",
#                     neigen=n_eigen, show=False)
#
#     # Do mixture model msm building
#     rev_counts, t_matrix, populations, mapping = \
#                     buildmsm.build_from_memberships(memberships,
#                                                     lag_time=lag_time)
#     fcm.analyze_msm(t_matrix, centroids, "Mixture Model",
#                     neigen=n_eigen, show=True)
#
#     # Try to improve by using a HMM
#     hidden_mm = get_hidden_markov_model(first_mixture_model, t_matrix)
#     trajs = get_data.get_trajs()
#     new_t_matrix = perform_optimization(hidden_mm, trajs)
#
#     fcm.analyze_msm(new_t_matrix, centroids, "Baum Welch Mixture Model",
#                 neigen=n_eigen, show=True)

if __name__ == "__main__":
    print("Do not run this file directly")




