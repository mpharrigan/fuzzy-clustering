from fuzzy import buildmsm, fcm, get_data, mullerforce
from matplotlib import pyplot as pp
from sklearn import mixture
import numpy as np

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


def test_mixture(min_k=3, max_k=20, fix_k=None, n_eigen=4):
    points = get_data.get_points(stride=3)
    
    min_mm = get_mixture_model(points, min_k, max_k, fix_k)
    
    centroids = min_mm.means_
    fcm.plot_centroids(centroids)
    plot_distribution(min_mm)
    pp.show()
    
    # Get the memberships
    memberships = min_mm.predict_proba(points)
    
    # Pick highest membership, put it in a classic state transition list,
    # and use straight msmbuilder code to build the count matrix
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_classic_from_memberships(memberships)
    fcm.analyze_msm(t_matrix, centroids, desc='Old, Hard',
                    neigen=n_eigen, show=False)
    
    # Quantize the membership vectors and use the new method of building the
    # count matrix and test with above
    q_memberships = quantize_memberships(memberships)
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(q_memberships)
    fcm.analyze_msm(t_matrix, centroids, "New, Hard",
                    neigen=n_eigen, show=False)
    
    # Do mixture model msm building
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(memberships)
    fcm.analyze_msm(t_matrix, centroids, "Mixture Model",
                    neigen=n_eigen, show=False)
    
    
    return
    
    high_state_mm = get_mixture_model(points, min_k, max_k, fix_k=200)
    high_state_memberships = high_state_mm.predict_proba(points)
    high_state_centroids = high_state_mm.means_
    rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_from_memberships(high_state_memberships)
    fcm.analyze_msm(t_matrix, high_state_centroids, "High state Mixture Model",
                    neigen=n_eigen, show=False)    
    
if __name__ == "__main__":
    test_mixture()

    
    
