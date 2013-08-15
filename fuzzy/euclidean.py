from msmbuilder.metrics.baseclasses import Vectorized
import numpy as np
import pickle

def make(metric_fn):
    f = open(metric_fn, 'w')
    e = Euclidean2d()
    pickle.dump(e, f)

class Euclidean2d(Vectorized):

    def prepare_trajectory(self, trajectory):
        # xyz = trajectory.xyz
        xyz = trajectory["XYZList"]
        xyz[:, :, 2] = 0.0
        nframes, natoms, ndims = xyz.shape
        return xyz.reshape(nframes, natoms * ndims).astype(np.double)


