"""Use mixture modeling and HMM to build a transition matrix."""

from fuzzy import buildmsm, fcm, get_data, mullerforce
from matplotlib import pyplot as pp
from sklearn import mixture
import ghmm
import ghmmhelper
import ghmmwrapper
import numpy as np
import scipy.sparse.csr

def plot_distribution(mixture_model, n_contours=80):
    """Plot the mixture distribution."""

    minx, maxx, miny, maxy = mullerforce.get_default_bounds()
    grid_width = max(maxx - minx, maxy - miny) / 200.0
    xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]

    mixture_samples = np.c_[xx.ravel(), yy.ravel()]
    contour_data = mixture_model.score(mixture_samples)
    contour_data = -contour_data.reshape(xx.shape)

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


def get_mixture_model(points, min_k, max_k, fix_k=None):
    """Find the best mixture model based on BIC."""
    prev_mm = None
    prev_bic = None
    min_mm = None

    if fix_k is None:
        for k in xrange(min_k, max_k):
            mm = mixture.GMM(n_components=k, covariance_type='full')
            mm.fit(points)

            bic = mm.bic(points)
            print("Trying k = %d, BIC = %g" % (k, bic))

            if prev_bic is not None and bic > prev_bic:
                min_mm = prev_mm
                break

            prev_bic = bic
            prev_mm = mm
            min_mm = mm
        del mm
    else:
        min_mm = mixture.GMM(n_components=fix_k, covariance_type='full')
        min_mm.fit(points)

    print_ic(min_mm, points)
    return min_mm

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

def perform_optimization(hidden_mm, trajs, n_steps=1000):
    """Optimize a hidden markov model given a list of trajectories.

    Use the Baum-Welch algorithm for learning the transition matrix, fixing
    emission probabilities.
    """

    # Domains for our multivariate gaussians
    domain = hidden_mm.emissionDomain

    # Prepare the trajectories by flattening them to 1D
    prepared_trajs = [t.flatten().tolist() for t in trajs]
    # Build the c-style sequences object manually
    (seq_c, lengths) = ghmmhelper.list2double_matrix(prepared_trajs)
    lengths_c = ghmmwrapper.list2int_array(lengths)
    cseq = ghmmwrapper.ghmm_cseq(seq_c, lengths_c, len(trajs))

    # Make a SequenceSet wrapper around the c-style object
    train_seq = ghmm.SequenceSet(domain, cseq)
    # Perform the Baum Welch optimization
    likelihood = hidden_mm.baumWelch(train_seq, nrSteps=n_steps)

    # Get the transition matrix in the normal form
    new_t_matrix = hidden_mm.asMatrices()[0]
    new_t_matrix = scipy.sparse.csr_matrix(new_t_matrix)

    return new_t_matrix

def hmm(traj_list, min_k=3, max_k=20, lag_time=1, n_eigen=4, show=False):
    """Self-contained method to perform full HMM analysis."""
    points = get_data.get_points_from_trajlist(traj_list)
    mixture_model = get_mixture_model(points, min_k, max_k)
    centroids = mixture_model.means_
    memberships = mixture_model.predict_proba(points)

    # Build an initial MSM as a guess
    rev_counts, t_matrix, populations, mapping = \
                buildmsm.build_from_memberships(memberships, lag_time=lag_time)
    fcm.analyze_msm(t_matrix, mixture_model.means_, "Mixture Model Guess",
                    neigen=n_eigen, show=False)

    # Learn from trajectories in HMM
    hidden_mm = get_hidden_markov_model(mixture_model, t_matrix)
    t_matrix = perform_optimization(hidden_mm, traj_list)

    fcm.analyze_msm(t_matrix, centroids, "Baum Welch HMM",
                neigen=n_eigen, show=show)

    return t_matrix

def test_mixture(min_k=3, max_k=20, fix_k=None, n_eigen=4, lag_time=10):
    """Run a bunch of functions to test the various schemes."""

    points = get_data.get_points(stride=1)
    mixture_model = get_mixture_model(points, min_k, max_k, fix_k)

    centroids = mixture_model.means_
    fcm.plot_centroids(centroids)
    plot_distribution(mixture_model)
    # pp.show()

    # Get the memberships
    memberships = mixture_model.predict_proba(points)

    # Pick highest membership, put it in a classic state transition list,
    # and use straight msmbuilder code to build the count matrix
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_classic_from_memberships(memberships,
                                                            lag_time=lag_time)
    fcm.analyze_msm(t_matrix, centroids, desc='Old, Hard',
                    neigen=n_eigen, show=False)

    # Quantize the membership vectors and use the new method of building the
    # count matrix and test with above
    q_memberships = quantize_memberships(memberships)
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(q_memberships,
                                                    lag_time=lag_time)
    fcm.analyze_msm(t_matrix, centroids, "New, Hard",
                    neigen=n_eigen, show=False)

    # Do mixture model msm building
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(memberships,
                                                    lag_time=lag_time)
    fcm.analyze_msm(t_matrix, centroids, "Mixture Model",
                    neigen=n_eigen, show=True)

    # Try to improve by using a HMM
    hidden_mm = get_hidden_markov_model(mixture_model, t_matrix)
    trajs = get_data.get_trajs()
    new_t_matrix = perform_optimization(hidden_mm, trajs)

    fcm.analyze_msm(new_t_matrix, centroids, "Baum Welch Mixture Model",
                neigen=n_eigen, show=True)

def main():
    trajs = get_data.get_trajs()
    hmm(trajs, lag_time=10)

if __name__ == "__main__":
    # test_mixture()
    main()




