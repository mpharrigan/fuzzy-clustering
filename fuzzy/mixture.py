from fuzzy import buildmsm, fcm, get_data, mullerforce
from matplotlib import pyplot as pp
from sklearn import mixture
import numpy as np
import ghmm
import scipy.sparse.csr
import ghmmwrapper
import ghmmhelper

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
    print("%s\tfit results: AIC: %g\tBIC: %g" % 
          (desc, mixture_model.aic(points), mixture_model.bic(points)))
    

def quantize_memberships(memberships):
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
    prev_mm = None
    prev_bic = None
    min_mm = None
    
    if fix_k is None:    
        for k in xrange(min_k, max_k):
            g = mixture.GMM(n_components=k, covariance_type='full')
            g.fit(points)
            
            bic = g.bic(points)
            print("Trying k = %d, BIC = %g" % (k, bic))
            
            if prev_bic is not None and bic > prev_bic:
                min_mm = prev_mm
                break
                
            prev_bic = bic
            prev_mm = g
            min_mm = g
        del g
    else:
        min_mm = mixture.GMM(n_components=fix_k, covariance_type='full')
        min_mm.fit(points)        
    
    print_ic(min_mm, points)
    return min_mm

def get_hidden_markov_model(mixture_model, guess_t_matrix):
    
    # Emission  probabilities for HMM
    emissions = [[mixture_model.means_[j], mixture_model.covars_[j].flatten()] for j in xrange(mixture_model.n_components)]
    
    # Initial transition matrix
    if isinstance(guess_t_matrix, scipy.sparse.csr.csr_matrix):
        guess_t_matrix = guess_t_matrix.todense()
        guess_t_matrix = guess_t_matrix.tolist()
        
    # Initial occupancy
    # Todo: figure out if initial occupancy matters
    initial_occupancy = [1.0 / mixture_model.n_components] * mixture_model.n_components
        
    # Set up distribution
    g_float = ghmm.Float()
    g_distribution = ghmm.MultivariateGaussianDistribution(g_float)    
    
    
    model = ghmm.HMMFromMatrices(g_float, g_distribution, guess_t_matrix, emissions, initial_occupancy)
    return model

def perform_optimization(hidden_mm, trajs, n_steps=1000):
    
    dim = trajs[0].shape[1]
    
    # Domains for our multivariate gaussians
    domain = hidden_mm.emissionDomain
    
    prepared_trajs = [t.flatten().tolist() for t in trajs]
    (seq_c, lengths) = ghmmhelper.list2double_matrix(prepared_trajs)
    lengths_c = ghmmwrapper.list2int_array(lengths)
    cseq = ghmmwrapper.ghmm_cseq(seq_c, lengths_c, len(trajs))
    
#     # Make our own c-style sequence object and bypass python wrapping (sortof)
#     points_flat = points.flatten()
#     points_flat_c = ghmmwrapper.list2double_array(points_flat)
#     cseq = ghmmwrapper.ghmm_cseq(points_flat_c, len(points_flat))
#     cseq.dim = dim
#     
#     train_seq = ghmm.EmissionSequence(domain, cseq)
     

    train_seq = ghmm.SequenceSet(domain, cseq)
    likelihood = hidden_mm.baumWelch(train_seq, nrSteps=n_steps)    
    
    new_t_matrix = hidden_mm.asMatrices()[0]
    new_t_matrix = scipy.sparse.csr_matrix(new_t_matrix)
    
    return new_t_matrix 

def test_mixture(min_k=3, max_k=20, fix_k=None, n_eigen=4, lag_time=10):
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
                    buildmsm.build_classic_from_memberships(memberships, lag_time=10)
    fcm.analyze_msm(t_matrix, centroids, desc='Old, Hard',
                    neigen=n_eigen, show=False)
    
    # Quantize the membership vectors and use the new method of building the
    # count matrix and test with above
    q_memberships = quantize_memberships(memberships)
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(q_memberships, lag_time=10)
    fcm.analyze_msm(t_matrix, centroids, "New, Hard",
                    neigen=n_eigen, show=False)
    
    # Do mixture model msm building
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(memberships, lag_time=10)
    fcm.analyze_msm(t_matrix, centroids, "Mixture Model",
                    neigen=n_eigen, show=True)
    
    # Try to improve by using a HMM
    hidden_mm = get_hidden_markov_model(mixture_model, t_matrix)
    trajs = get_data.get_trajs()
    new_t_matrix = perform_optimization(hidden_mm, trajs)
    
    fcm.analyze_msm(new_t_matrix, centroids, "Baum Welch Mixture Model",
                neigen=n_eigen, show=True)
    
    return points, mixture_model, memberships, t_matrix, hidden_mm
    
    high_state_mm = get_mixture_model(points, min_k, max_k, fix_k=200)
    high_state_memberships = high_state_mm.predict_proba(points)
    high_state_centroids = high_state_mm.means_
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(high_state_memberships)
    fcm.analyze_msm(t_matrix, high_state_centroids, "High state Mixture Model",
                    neigen=n_eigen, show=False)    
    
if __name__ == "__main__":
    test_mixture()

    
    
