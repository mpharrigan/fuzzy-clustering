from __future__ import division
import numpy as np
from fuzzy import analysis, classic, mixture
import msmtoys.analytic
import pickle
import itertools
from matplotlib import pyplot as pp


def main(tmatrix_fn='transition_matrix_70.pickl', num_trajs=10, traj_len=500000, stride=1, n_its=6, results_fn='validation_results_3.pickl'):
    """Call the calculation steps in order and write the results to disk."""

    validator = Validator(tmatrix_fn, num_trajs, traj_len, stride, n_its)
    validator.do_validation()


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
        self.grid = grid

        self.num_trajs = num_trajs
        self.traj_len = traj_len
        self.stride = stride
        self.n_its = n_its

        param_random_seeds = range(10)
        param_nclusters = range(2, 20)
        param_lagtimes = range(1, 30, 2)

        # Debug
        param_random_seeds = range(2)
        param_nclusters = [2, 4]
        param_lagtimes = [1, 10]

        self.vd = ValidationData(n_its, param_random_seeds, param_nclusters, param_lagtimes)

        print "Using Aggregate: {:,} points".format((num_trajs * traj_len / stride))

        self.traj_list = None

    def new_trajlist(self):
        self.traj_list = msmtoys.analytic.get_trajlist(self.t_matrix, self.grid, self.num_trajs, self.traj_len, self.stride)

    def calculate_hmm(self):
        """Calculate results for HMM."""
        def tmat_func(traj_list, *params):
            _, new_t_matrix, _, _, _ = mixture.hmm(traj_list, fix_k=params[1], lag_time=params[2], sliding_window=True, mm_stride=50)
            return new_t_matrix
        def anal_func(t_matrix, *params):
            return analysis.get_implied_timescales(t_matrix, n_timescales=self.n_its, lag_time=params[2] * self.stride)
        def set_func(param_is, its):
            self.vd.hmm_its[param_is] = its

        return tmat_func, anal_func, set_func

    def calculate_msm(self):
        """Calculate results for MSM."""
        def tmat_func(traj_list, *params):
            # NOTE: I am accounting for number of parameter scaling here by multiplying number of clusters by 3
            return classic.msm(traj_list, n_clusters=params[1] * 3, n_medoid_iters=10, lag_time=params[2])
        def anal_func(t_matrix, *params):
            return analysis.get_implied_timescales(t_matrix, n_timescales=self.n_its, lag_time=params[2] * self.stride)
        def set_func(param_is, its):
            self.vd.msm_its[param_is] = its

        return tmat_func, anal_func, set_func


    def calculate_analytic_its(self):
        """Calculate analytic results."""
        analytic_its = analysis.get_implied_timescales(self.t_matrix, n_timescales=self.n_its, lag_time=1)
        self.vd.analytic_its = analytic_its

    def calculate_its(self, tmat_funcs, anal_funcs, set_funcs):
        """Calculate implied timescales given appropriate functions.

        This function calls ``vd.get_param_iter()`` to get the parameter
        indices over which to iterate and apply the provided functions.

        :param tmat_funcs: list of ``tmat_func(traj_list, *params)`` that returns a
                            transition matrix given parameters
        :param anal_funcs: list of ``anal_func(t_matrix, *params)`` that returns implied
                            timescales given a transition matrix and parameters
        :param set_funcs: list of ``set_func(param_is, its)`` that is responsible for
                                        saving the results, ``its``, at
                                        parameter indices ``param_is``
        """
        assert len(tmat_funcs) == len(anal_funcs)
        assert len(anal_funcs) == len(set_funcs)
        progress = 0

        for repeat_i in self.vd.get_repeats():
            self.new_trajlist()
            for param_is in self.vd.get_param_iter():
                # Add in repeat index
                pi = list(param_is)
                pi.insert(0, repeat_i)
                param_is = tuple(pi)

                param_values = self.vd.translate_to_values(param_is)
                print "Using parameter values", param_values
                print "Progress: %d/%d = %.2f%%" % (progress, self.vd.n_permuts, 100.0 * progress / self.vd.n_permuts)


                for tmat_func, anal_func, set_func in zip(tmat_funcs, anal_funcs, set_funcs):
                    t_matrix = tmat_func(self.traj_list, *param_values)
                    its = anal_func(t_matrix, *param_values)
                    set_func(param_is, its)

                progress += 1

    def do_validation(self):
        htmat, hanal, hset = self.calculate_hmm()
        mtmat, manal, mset = self.calculate_msm()
        self.calculate_its([htmat, mtmat], [hanal, manal], [hset, mset])



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
        self.n_permuts = np.product(its_shape[1:])

        its_shape.append(n_its)
        self.hmm_its = np.zeros(tuple(its_shape))
        self.msm_its = np.zeros(tuple(its_shape))

        self.param_string = None

    def get_param_iter(self):
        """Make an iterator over all permutations of parameters."""
        return itertools.product(*[xrange(len(param)) for param in self.params[1:]])

    def get_repeats(self):
        return xrange(len(self.params[0]))

    def translate_to_values(self, param_is):
        """Take a tuple of indices and turn it into a list of values."""
        param_values = list()
        for t in xrange(len(param_is)):
            param_i = param_is[t]
            param_values.append(self.params[t][param_i])
        return param_values

    def plot(self, param_is, it_i, show=3):
        assert len(param_is) == len(self.params)

        vary_i = None
        for i in xrange(len(param_is)):
            if param_is[i] is None:
                vary_i = i
                break

        xs = list()
        for i in xrange(len(self.params[vary_i])):
            param_is[vary_i] = i
            x = self.translate_to_values(tuple(param_is))[vary_i]
            xs.append(x)

        param_is[vary_i] = slice(None, None, None)

        if show == 1 or show == 3:
            ys = self.hmm_its[tuple(param_is) + (it_i,)]
            pp.plot(xs, ys, label='HMM')

        if show == 2 or show == 3:
            ys = self.msm_its[tuple(param_is) + (it_i,)]
            pp.plot(xs, ys, label='MSM')

        pp.yscale('log')
        pp.legend()



if __name__ == "__main__":
    main()
