from __future__ import division
import numpy as np
from fuzzy import analysis, classic, mixture
import msmtoys.analytic
import pickle
import itertools


def main(tmatrix_fn='transition_matrix.pickl', num_trajs=100, traj_len=5000, stride=10, n_its=6, results_fn='validation_results.pickl'):
    """Call the calculation steps in order and write the results to disk."""

    validator = Validator(tmatrix_fn, num_trajs, traj_len, stride, n_its)

    print "Calculating analytic ITs..."
    validator.calculate_analytic_its()

    print "Calculating HMM ITs..."
    validator.calculate_hmm()

    print "Calculating MSM ITs..."
    validator.calculate_msm()

    print "Saving results..."
    with open(results_fn, 'w') as results_f:
        pickle.dump(validator.vd, results_f)

class Validator():
    def __init__(self, tmatrix_fn, num_trajs, traj_len, stride, n_its):
        """Initialize the validation protocol.

        During initialization, the transition matrix is loaded and parameters
        are defined. A `ValidationData` object is also created, which
        allocates space for results.
        """
        with open(tmatrix_fn, 'rb') as tmatrix_f:
            t_matrix, grid = pickle.load(tmatrix_f)

        self.t_matrix = t_matrix

        self.num_trajs = num_trajs
        self.traj_len = traj_len
        self.stride = stride
        self.n_its = n_its

        param_random_seeds = range(52, 63)
        param_nclusters = range(2, 20)
        param_lagtimes = range(1, 30, 2)

        self.vd = ValidationData(n_its, param_random_seeds, param_nclusters, param_lagtimes)

        print "Using Aggregate: {:,} points".format((num_trajs * traj_len / stride))
        def trajlist_func(*params):
            return msmtoys.analytic.get_trajlist(t_matrix, grid, num_trajs, traj_len, stride, random_seed=params[0])

        self.trajlist_func = trajlist_func

    def calculate_hmm(self):
        """Calculate results for HMM."""
        def tmat_func(traj_list, *params):
            return mixture.hmm(traj_list, fix_k=params[1], lag_time=params[2], sliding_window=True)[0]
        def anal_func(t_matrix, *params):
            return analysis.get_implied_timescales(t_matrix, n_timescales=self.n_its, lag_time=params[2] * self.stride)
        def set_func(param_is, its):
            self.vd.hmm_its[param_is] = its

        self.calculate_its(tmat_func, anal_func, set_func)

    def calculate_msm(self):
        """Calculate results for MSM."""
        def tmat_func(traj_list, *params):
            return classic.msm(traj_list, n_clusters=params[1], n_medoid_iters=10, lag_time=params[2])
        def anal_func(t_matrix, *params):
            return analysis.get_implied_timescales(t_matrix, n_timescales=self.n_its, lag_time=params[2] * self.stride)
        def set_func(param_is, its):
            self.vd.msm_its[param_is] = its

        self.calculate_its(tmat_func, anal_func, set_func)


    def calculate_analytic_its(self):
        """Calculate analytic results."""
        analytic_its = analysis.get_implied_timescales(self.t_matrix, n_timescales=self.n_its, lag_time=1)
        self.vd.analytic_its = analytic_its

    def calculate_its(self, tmat_func, anal_func, set_func):
        """Calculate implied timescales given appropriate functions.

        This function calls ``vd.get_param_iter()`` to get the parameter
        indices over which to iterate and apply the provided functions.

        :param tmat_func: ``tmat_func(traj_list, *params)`` that returns a
                            transition matrix given parameters
        :param anal_func: ``anal_func(t_matrix, *params)`` that returns implied
                            timescales given a transition matrix and parameters
        :param set_func: ``set_func(param_is, its)`` that is responsible for
                                        saving the results, ``its``, at
                                        parameter indices ``param_is``
        """
        progress = 0
        for param_is in self.vd.get_param_iter():
            param_values = self.vd.translate_to_values(param_is)
            print "Using parameter values", param_values
            print "Progress: %d/%d = %.2f%%" % (progress, self.vd.n_permuts, 100.0 * progress / self.vd.n_permuts)

            traj_list = self.trajlist_func(*param_values)
            t_matrix = tmat_func(traj_list, *param_values)
            its = anal_func(t_matrix, *param_values)
            set_func(param_is, its)

            progress += 1


class ValidationData:
    """Container for our various results."""

    def __init__(self, n_its, *params):
        """Initializing this class allocates arrays and produces parameter
        permutations.

        :param n_its: The number of implied timescales for which to allocate
                      space
        :type n_its: int
        :param *params: each param is a list of values over which to test.
        """
        self.n_its = n_its
        self.params = params
        self.analytic_its = np.zeros(n_its)

        its_shape = [len(param) for param in params]
        self.n_permuts = np.product(its_shape)

        its_shape.append(n_its)
        self.hmm_its = np.zeros(tuple(its_shape))
        self.msm_its = np.zeros(tuple(its_shape))

    def get_param_iter(self):
        """Make an iterator over all permutations of parameters."""
        return itertools.product(*[xrange(len(param)) for param in self.params])

    def translate_to_values(self, param_is):
        """Take a tuple of indices and turn it into a list of values."""
        param_values = list()
        for t in xrange(len(param_is)):
            param_i = param_is[t]
            param_values.append(self.params[t][param_i])
        return param_values


if __name__ == "__main__":
    main()
