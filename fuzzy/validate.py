from __future__ import division
import numpy as np
from msmtoys import muller, plotting
from fuzzy import analysis, classic, mixture
import msmtoys.analytic
import pickle

def validate(tmatrix_fn):
    with open(tmatrix_fn, 'rb') as tmatrix_f:
        t_matrix, grid = pickle.load(tmatrix_f)
        
    vd = ValidationData()
        
    print "Aggregate: {:,} points".format((num_trajs*traj_len/stride))
    traj_list = msmtoys.analytic.get_trajlist(t_matrix, grid, num_trajs, traj_len, stride)
    
    analytic_its = analysis.get_implied_timescales(t_matrix, n_timescales=n_timescales, lag_time=1)
    
    
    
class ValidationData:
    """Container for all the stuffs."""
    
    def __init__(self):
        self.analytic_its = None