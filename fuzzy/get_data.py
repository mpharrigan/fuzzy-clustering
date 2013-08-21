from __future__ import division

import mdtraj
import os
import numpy as np


def get_trajs(directory="../test_trajs/", dim=2, retrieve='justpoints'):
    """Get a list of trajectories.
    
    Use retrieve to determine type:
     - 'justpoints': x and y coordinates of the first particle in a numpy
                     array, list of
     - 'mdtrajs':    mdtraj objects, list of
     - 'shimtrajs':  trajectories suitable for use with msmbuilder2
     """
    files = os.listdir(directory)
    trajlist = list()
    for f in files:
        if f.endswith('.h5'):
            traj = mdtraj.load("%s/%s" % (directory, f))
                        
            if retrieve == 'justpoints':
                # Get x and y of the first particle
                xy = traj.xyz[:, 0, 0:dim]
                
                # Add them to the list 
                trajlist.append(xy)
            elif retrieve == 'mdtrajs':
                trajlist.append(traj)
            elif retrieve == 'shimtrajs':
                shimt = ShimTrajectory(traj.xyz)
                trajlist.append(shimt)
            
    
    return trajlist

def get_points_from_trajlist(traj_list):
    """Get a list of points from multiple trajectories.
    
    This can be used for clustering, mixture modeling, etc but is
    inappropriate for e.g. training an HMM.
    """
    assert len(traj_list) > 0, 'Please supply at least one trajectory'
    dim = traj_list[0].shape[1]
    points = np.zeros((0, dim))
    for traj in traj_list:
        points = np.append(points, traj, axis=0)
    return points
        

def get_points(stride, directory="../test_trajs/", dim=2):
    """Returns a numpy array of xy points."""
    files = os.listdir(directory)
    points = np.zeros((0, dim))
    for f in files:
        if f.endswith('.h5'):
            traj = mdtraj.load("%s/%s" % (directory, f))
                        
            # Get x and y of the first particle
            xy = traj.xyz[:, 0, 0:dim]
            
            # Add them to the list 
            points = np.append(points, xy, axis=0)
    
    
    n_points = len(points)
    points = points[::stride    , :]
    n_points_left = len(points)
    print("Loaded %d points. Using %d (%f %%)" % (n_points, n_points_left, 100 * n_points_left / n_points))
    
    return points    

class ShimTrajectory(dict):
    """This is a dict that can be used to interface some xyz coordinates
    with MSMBuilder's clustering algorithms.

    I'm really sorry that this is necessary. It's horridly ugly, but it comes
    from the fact that I want to use the mdtraj trajectory object (its better),
    but the msmbuilder code hasn't been rewritted to use the mdtraj trajectory
    yet. Soon, we will move mdtraj into msmbuilder, and this won't be necessary.
    """
    def __init__(self, xyz):
        super(ShimTrajectory, self).__init__()
        self['XYZList'] = xyz

    def __len__(self):
        return len(self['XYZList'])
    
    
