import numpy as np
from sklearn import mixture
import get_data
import fcm
from matplotlib import pyplot as pp
import mullerforce
import buildmsm

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
    print("%s\tfit results: AIC: %g\tBIC: %g" % (desc, mixture_model.aic(points), mixture_model.bic(points)))
    


def test_mixture(min_k=3, max_k=20, fix_k=None):
    points = get_data.get_points(stride=10)
    
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
    
    centroids = min_mm.means_
    fcm.plot_centroids(centroids)
    plot_distribution(min_mm)
    pp.show()
    
    memberships = min_mm.predict_proba(points)
    rev_counts, t_matrix, populations, mapping = buildmsm.build_from_memberships(memberships)
    fcm.analyze_msm(t_matrix, centroids, "Mixture Model", neigen=3, show=True)
    

    
    
