"""Build an MSM using classic methods."""

from fuzzy import get_data
from msmbuilder import MSMLib as msml, clustering
from msmbuilder.metrics.baseclasses import Vectorized
import numpy as np


def cluster(traj_list, n_clusters, n_medoid_iters=10, distance_cutoff=None, dim=3):
    """Do medoids clustering

    :param dim: ShimTrajectory will be padded to this dimension.
    """
    if distance_cutoff is not None:
        n_clusters = None

    trajs = get_data.get_shimtraj_from_trajlist(traj_list, dim)
    metric = Euclidean2d()

    clustering.logger.setLevel('WARNING')
    hkm = clustering.HybridKMedoids(metric, trajs, k=n_clusters, distance_cutoff=distance_cutoff, local_num_iters=n_medoid_iters)


    return hkm

def msm(traj_list, n_clusters, n_medoid_iters=10, lag_time=1, distance_cutoff=None):
    """Use classic clustering methods."""

    hkm = cluster(traj_list, n_clusters, n_medoid_iters, distance_cutoff)
    # centroids = hkm.get_generators_as_traj()
    # centroids_nf = centroids['XYZList'][:, 0, 0:dim]

    counts = msml.get_count_matrix_from_assignments(hkm.get_assignments(), n_clusters, lag_time)
    rev_counts, t_matrix, populations, mapping = msml.build_msm(counts)

    return t_matrix


class Euclidean2d(Vectorized):

    def prepare_trajectory(self, trajectory):
        # xyz = trajectory.xyz
        xyz = trajectory["XYZList"]
        if xyz.shape[2] == 3:
            # TODO: Make this better
            xyz[:, :, 2] = 0.0
        nframes, natoms, ndims = xyz.shape
        return xyz.reshape(nframes, natoms * ndims).astype(np.double)
