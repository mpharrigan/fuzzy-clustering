from matplotlib import pyplot as pp
from msmbuilder import MSMLib as msml, msm_analysis as msma, clustering
import fuzzy.buildmsm as buildmsm
import fuzzy.euclidean as euclidean
import fuzzy.get_data as get_data
import fuzzy.mullerforce as mullerforce
import numpy as np
import random
import scipy.sparse
    
    
def _membership_frac(point_j, centroid_i, fuzziness, dist):
    """A term in computing membership."""
    fraction = 1.0 / dist(point_j, centroid_i)
    exponent = 1.0 / (fuzziness - 1)
    value = fraction ** exponent
    return value
    
def calculate_membership(i, j, points, centroids, fuzziness, dist, debug):
    """Calculate the membership u_ij given data points and centroids."""
    
    n_clusters = len(centroids)
    
    numerator = _membership_frac(points[j], centroids[i], fuzziness, dist)
    denomonat = 0.0
    for k in xrange(n_clusters):
        denomonat += _membership_frac(points[j], centroids[k], fuzziness, dist)
        
    result = numerator / denomonat
    
    if debug:
        print("Membership (%d,%d) is %f" % (i, j, result))
    
    return result

def calculate_memberships(points, centroids, fuzziness, dist, debug=False):
    """Calculate memberships of points relative to centroids."""
    n_points = len(points)
    n_clusters = len(centroids)
    
    memberships = np.zeros((n_clusters, n_points))
    
    for j in xrange(n_points):
        for i in xrange(n_clusters):
            memberships[i, j] = calculate_membership(i, j, points, centroids,
                                                     fuzziness, dist, debug)
            
    return memberships
            
def calculate_centroid(i, points, memberships, fuzziness):
    """Calculates the centroid V_i"""
    n_points = len(points)
    
    numerator = 0.0
    denomonat = 0.0
    for j in xrange(n_points):
        term = memberships[i, j] ** fuzziness
        numerator += term * points[j]
        denomonat += term
        
    result = numerator / denomonat
    return result
    
def calculate_centroids(k, dim, points, memberships, fuzziness):
    """Calculates k centroids relative to points and their memberships."""
    centroids = np.zeros((k, dim))
    
    for i in xrange(k):
        centroids[i] = calculate_centroid(i, points, memberships, fuzziness)
        
    return centroids

def euclidean_distance(x1, x2):
    """The euclidean distance."""
    difference = x2 - x1
    norm = np.dot(difference, difference)
    norm = np.sqrt(norm)
    return norm

def guess_centroids_smart(k):
    """A cheat function that makes good guesses for the centroids of the muller
    potential.
    
    Values are hard-coded. If more than k=4 centroids are requested, this
    function returns None and it is up to the calling function to provide a
    fallback mechanism.
    """
    centroids = np.array([
                          [0.486, 0.0995],
                          [-0.247, 0.3749],
                          [-1.014, 0.66],
                          [-0.66, 1.25]
                          ])
    if k > len(centroids):
        return None
    return centroids[:k]

def guess_centroids(k, dim=2, bounds=None, smart_guess=False):
    """Provide an initial guess for centroids.
    
    if smart_guess is True, this function will attempt to 'cheat' and use
    hardcoded centroid positions. If more than k=4 smartly-guessed centroids
    are requested, it will panic and just return random centroids.
    
    Otherwise, random centroids (within the bounds of the muller potential)
    are generated.
    """
    if smart_guess:
        centroids = guess_centroids_smart(k)
        if centroids is not None:
            return centroids
        else:
            print("k is too high, defaulting to random centroid guess")
    
    if bounds is None:
        bounds = mullerforce.get_default_bounds()
    
    print("Guessing initial centroids in %s" % bounds.__str__())
    
    centroids = np.zeros((k, dim))
    
    for i in xrange(k):
        for j in xrange(dim):
            rand = ((random.random() * (bounds[2 * j + 1] - bounds[2 * j ])) 
                    + bounds[2 * j ])
            centroids[i, j] = rand
            
    return centroids

def generate_test_points(n_points_per=100, dim=2, n_centers=3, bounds=None,
                         variance=0.6):
    """Generate random points over the muller potential."""
    if bounds is None:
        bounds = mullerforce.get_default_bounds()
        
    temp_center = np.zeros(dim)
    points = np.zeros((n_points_per * n_centers, dim))
    
    for i in xrange(n_centers):
        # For each center, pick a center point
        for k in xrange(dim):
            rand = ((random.random() * (bounds[2 * k + 1] - bounds[2 * k ]))
                    + bounds[2 * k ])
            temp_center[k] = rand
        
        # For each center, having computed a center point, make a bunch of
        # points near that center
        for j in xrange(n_points_per):
            for k in xrange(dim):
                rand = random.random() * variance   
                points[i * n_points_per + j, k] = temp_center[k] + rand
                    
    return points
            
def convergence(old_memberships, new_memberships, epsilon):
    """Test convergence for fuzzy clustering."""
    max_diff = np.max(np.abs(old_memberships - new_memberships))
    return max_diff < epsilon    
    
def plot_centroids(centroids, marker_sizes=None):
    """Plot the centroids of a particular clustering scheme.
    
    If marker_sizes is given, it will use this array as the sizes for the
    various centroids. This is useful for visualizing eigenvectors.
    """
    if marker_sizes is None:
        pp.clf()
        mullerforce.MullerForce.plot()
        pp.plot(centroids[:, 0], centroids[:, 1], 'yo', markersize=12,
                zorder=10)
        pp.title("Centroids")
    else:
        assert len(centroids) == len(marker_sizes)
        for i in xrange(len(centroids)):
            marker_size = marker_sizes[i]
            if marker_size < 0:
                marker_size = np.abs(marker_size)
                color_string = 'ro'
            else:
                color_string = 'wo'
            pp.plot(centroids[i, 0], centroids[i, 1], color_string,
                    markersize=12 * 3 * marker_size, zorder=10)
    
def plot_points(points):
    """Plot points."""
    pp.plot(points[:, 0], points[:, 1], 'yo')
    
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
                   [204, 27, 224],  # Purple
                   ]
    base_colors = np.array(base_colors) / 255.
    
    memberships = memberships.transpose()
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
        colors[i, 3] = max_occupation
        
    pp.scatter(points[:, 0], points[:, 1], c=colors)
    
def get_hard_state_list(centroids, traj, fuzziness, dist):
    """From a trajectory and a clustering scheme, get the list of states
    through which the trajectory traverses in a 'hard' clustering scheme.
    
    This function uses the cluster of highest membership to describe
    state transitions.
    
    This function returns both a classic list of state indexes as well as
    a list of membership vectors that has been 'hardened' to be of the form
    [0, ..., 0, 1, 0, ..., 0].
    """
    memberships = calculate_memberships(traj, centroids, fuzziness, dist).transpose()
    
    state_list_classic = list()
    state_list_new = list()
    
    for memb in memberships:
        z = np.zeros(len(memb))
        state = np.argmax(memb)
        z[state] = 1.0
        state_list_classic.append(state)
        state_list_new.append(z)
            
    return np.array(state_list_classic), state_list_new

def get_soft_state_list(centroids, traj, fuzziness, dist):
    """From a trajectory and a clustering scheme, get the list of membership
    vectors through which the trajectory traverses."""
    memberships = calculate_memberships(traj, centroids, fuzziness, dist).transpose()
    
    state_list = list()    
    for memb in memberships:
        state_list.append(memb)        
            
    return state_list
    
def get_giant_state_list(centroids, trajs, fuzziness, dist, soft=True, lag_time=1):
    """Get a matrix of pairs of membership vectors.
    
    The resulting matrix has dimension n_points x 2 x n_clusters. The second
    index is either 0 or 1 and corresponds to 'from' or 'to' membership
    vectors.
    """
    n_times = 2
    n_clusters = len(centroids)
    time_pairs = np.zeros((0, 2, n_clusters))
    
    for traj in trajs:
        
        if soft:
            state_list = get_soft_state_list(centroids, traj, fuzziness, dist)
        else:
            _, state_list = get_hard_state_list(centroids, traj, fuzziness, dist)
        
        from_states = np.array(state_list[:-lag_time: lag_time])
        to_states = np.array(state_list[lag_time:: lag_time])
        
        assert len(from_states) == len(to_states)
        
        
        n_points = len(from_states)
        pairs = np.zeros((n_points, n_times, n_clusters))
        pairs[:, 0, :] = from_states
        pairs[:, 1, :] = to_states 
        
        time_pairs = np.append(time_pairs, pairs, axis=0)
        
    return time_pairs


def plot_eigens(centroids, vec, vals, desc):
    """Plot eigenvectors (one at a time)."""
    pp.clf()
    for i in xrange(len(vals)):
        mullerforce.MullerForce.plot()
        plot_centroids(centroids, vec[:, i])
        pp.title('Eigenvector %d, Lambda=%f' % ((i + 1), vals[i]))
        pp.suptitle(desc)
        pp.show()            

        
def analyze_msm(t_matrix, centroids, desc, neigen=4, show=False):
    """Analyze a particular msm.
    
    Right now, it does this by printing eigenvalues and optionally plotting
    eigenvectors.
    """
    val, vec = msma.get_eigenvectors(t_matrix, neigen)
    oolambda = -1.0 / np.log(val[1:])

    print("\n%s" % desc)
    print("Eigenvalues:\t%s" % val.__str__())
    print("1/lambda:\t%s" % oolambda.__str__())
    
    if show: plot_eigens(centroids, vec, val, desc)  

def build_new(centroids, trajs, fuzziness, dist, soft=True, neigen=4, show=False):
    """Build an MSM from membership vectors.
    
    if soft is False, 'Quantize' the membership vectors to mirror the
    hard clustering case, else use the fuzzy nature of the clusters in 
    building the MSM.
    """
    n_states = len(centroids)
    time_pairs = get_giant_state_list(centroids, trajs, fuzziness, dist, soft=soft)
    counts_mat = buildmsm.get_counts_from_pairs(time_pairs, n_states)
    rev_counts, t_matrix, populations, mapping = msml.build_msm(counts_mat)
    
    if soft:
        desc = 'New, Fuzzy'
    else:
        desc = 'New, not-so-fuzzy'
    analyze_msm(t_matrix, centroids, desc=desc, show=show)
    

def build_old(centroids, fuzziness, dist, neigen=4, show=False):
    """Build an MSM the old fashioned way."""
    # TODO: Change input to do trajs
    n_states = len(centroids)
    trajs = get_data.get_trajs()
    counts_classic = scipy.sparse.lil_matrix((int(n_states), int(n_states)),
                                             dtype='float32')
    
    for traj in trajs:
        sl_classic, _ = get_hard_state_list(centroids, traj, fuzziness, dist)
        counts_classic = counts_classic + \
                msml.get_counts_from_traj(sl_classic, n_states)        

    rev_counts, t_matrix, populations, mapping = msml.build_msm(counts_classic)  
    analyze_msm(t_matrix, centroids, desc='Old, Hard', show=show)
    

def fcm(cluster_points, trajs, fuzziness=2.0, dist=euclidean_distance, max_iter=50,
        eps=0.01, k=4, dim=2, show=False):
    """Do fuzzy clustering."""
    centroids = guess_centroids(k, dim=dim, smart_guess=True)
    memberships = calculate_memberships(cluster_points, centroids, fuzziness, dist)
    
    for i in xrange(max_iter):
        centroids = calculate_centroids(k, dim, cluster_points, memberships, fuzziness)
        new_memberships = calculate_memberships(cluster_points, centroids, fuzziness, dist)
        
        if convergence(memberships, new_memberships, eps):
            print("Convergence achieved after %d steps" % i)
            break
        
        memberships = new_memberships
    
    plot_centroids(centroids)
    plot_points_with_alpha(cluster_points, memberships)
    if show: pp.show()
    else: pp.clf()
    
    build_new(centroids, trajs, fuzziness, dist, soft=True, show=show)
    
def classic(trajs, n_clusters, n_medoid_iters, metric, dim=2, lag_time=1, show=False):
    """Use classic clustering methods."""
    hkm = clustering.HybridKMedoids(metric, trajs, n_clusters=n_clusters, local_num_iters=n_medoid_iters)
    centroids = hkm.get_generators_as_traj()
    
    centroids_nf = centroids['XYZList'][:, 0, 0:dim]
    plot_centroids(centroids_nf)
    if show: pp.show()
    
    counts = msml.get_count_matrix_from_assignments(hkm.get_assignments(), n_clusters, lag_time)
    rev_counts, t_matrix, populations, mapping = msml.build_msm(counts)
    analyze_msm(t_matrix, centroids_nf, 'Classic, n_clusters=%d' % n_clusters, show=show)

def demonstrate(show, classic_k=200, new_k=10, num_med_iters=1, lag_time=10):
    """Run through various schemes for comparison."""
    clustering.logger.setLevel('ERROR')

    # Load shimmed trajectories and a metric for interfacing with msmbuilder
    trajs = get_data.get_trajs(retrieve='shimtrajs')
    metric = euclidean.Euclidean2d()
    
    # Build msm in the classic regime (high number of clusters)
    classic(trajs, n_clusters=classic_k, n_medoid_iters=num_med_iters, metric=metric, lag_time=lag_time, show=True)
    
    # Build msm using classic methods with a small number of clusters
    classic(trajs, n_clusters=new_k, n_medoid_iters=num_med_iters, metric=metric, lag_time=lag_time, show=True)
    
    # Build msm using new fuzzy clusters
    points_sample = get_data.get_points(stride=10)
    trajs = get_data.get_trajs(retrieve='justpoints')
    fcm(points_sample, trajs, k=new_k, show=True)

if __name__ == "__main__":
    demonstrate(show=True)
    
