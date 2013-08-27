from __future__ import division
from msmtoys import analytic, muller, plotting
from fuzzy import mixture, fcm, validate
from matplotlib import pyplot as pp
from msmbuilder import msm_analysis as msma, clustering
from fuzzy import euclidean, get_data
import argparse
import pickle


def validate_hmm(analytic_tmatrix, grid, num_trajs, traj_len, stride):
    traj_list = get_trajlist(analytic_tmatrix, grid, num_trajs, traj_len, stride)
    hmm_tmatrix = mixture.hmm(traj_list, fix_k=3)

    # Classic
    clustering.logger.setLevel('ERROR')
    metric = euclidean.Euclidean2d()
    shim_t = get_data.get_shimtraj_from_trajlist(traj_list)
    classic_t_matrix = fcm.classic(shim_t, n_clusters=3, n_medoid_iters=10, metric=metric, lag_time=1, show=False)
    classic_lambda = fcm.analyze_msm(classic_t_matrix, None, desc="Classic", neigen=3, show=False)

    # Do big k classic
    big_classic_t_matrix = fcm.classic(shim_t, n_clusters=200, n_medoid_iters=10, metric=metric, lag_time=100, show=False)
    big_classic_lambda = fcm.analyze_msm(big_classic_t_matrix, None, desc="Classic (large k)", neigen=5, show=False)

    analytic_tmatrix_lambda = fcm.analyze_msm(analytic_tmatrix, None, desc="Analytic", neigen=5, show=False)
    hmm_tmatrix_lambda = fcm.analyze_msm(hmm_tmatrix, None, desc="HMM", neigen=5, show=False)

    validate.plot_lambda_bar([analytic_tmatrix_lambda, hmm_tmatrix_lambda, classic_lambda, big_classic_lambda], ['Analytic', 'HMM', 'Classic (small k)', 'Classic (larg k)'])
    pp.show()

def validate_hmm_main(args):
    num_trajs = args.num_trajs
    t_matrix, grid = unpickle_tmatrix(args)
    traj_len = args.traj_len
    stride = args.stride
    validate_hmm(t_matrix, grid, num_trajs, traj_len, stride)

def get_trajlist(t_matrix, grid, num_trajs, traj_len, stride):
    traj_list = list()
    for _ in xrange(num_trajs):
        traj = analytic.get_traj(t_matrix, length=traj_len, grid=grid, stride=stride)
        traj_list.append(traj)

    return traj_list


def show_trajectories_main(args):
    num_trajs = args.num_trajs
    t_matrix, grid = unpickle_tmatrix(args)
    traj_len = args.traj_len
    stride = args.stride

    traj_list = get_trajlist(t_matrix, grid, num_trajs, traj_len, stride)

    # Plot
    plotting.plot2d(muller.MullerForce)
    [pp.plot(traj[:, 0], traj[:, 1], '-') for traj in traj_list]
    pp.title("Sample trajectories")
    pp.show()

def unpickle_tmatrix(args):
    tmatrix_fn = args.tmatrix_fn
    with open(tmatrix_fn, 'rb') as tfile:
        t_matrix, grid = pickle.load(tfile)
    return t_matrix, grid

def calculate_tmat_main(args):
    tmatrix_fn = args.tmatrix_fn
    resolution = args.resolution
    beta = args.beta

    t_matrix, grid = analytic.calculate_transition_matrix(muller.MullerForce, resolution, beta)

    with open(tmatrix_fn, 'wb') as tfile:
        pickle.dump((t_matrix, grid), tfile)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', dest='tmatrix_fn',
                       help="""Transition matrix filename for pickle object""",
                       default='tmatrix.pickl')

    subparsers = parser.add_subparsers()

    calct = subparsers.add_parser('calct', help='Calculate transition matrix')
    calct.set_defaults(func=calculate_tmat_main)
    calct.add_argument('-b', dest='beta',
                       help='''Beta for transition matrix''', type=float,
                       default=1 / 70)
    calct.add_argument('-res', dest='resolution',
                       help='''Resolution of the transition matrix. Higher is
                       computationally intensive''', type=int, default=100)

    showtrajs = subparsers.add_parser('showtrajs', help="Show trajectories")
    showtrajs.set_defaults(func=show_trajectories_main)
    showtrajs.add_argument('-nt', dest='num_trajs',
                           help='''Number of trajectories to plot''',
                           type=int, default=5)
    showtrajs.add_argument('-lt', dest='traj_len',
                           help='''Length of trajectories''',
                           type=int, default=2000)
    showtrajs.add_argument('-s', dest='stride',
                           help='''Trajectory stride''',
                           type=int, default=10)

    validate = subparsers.add_parser('validate', help="Validate HMM")
    validate.set_defaults(func=validate_hmm_main)
    validate.add_argument('-nt', dest='num_trajs',
                           help='''Number of trajectories to plot''',
                           type=int, default=5)
    validate.add_argument('-lt', dest='traj_len',
                           help='''Length of trajectories''',
                           type=int, default=2000)
    validate.add_argument('-s', dest='stride',
                           help='''Trajectory stride''',
                           type=int, default=10)


    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    parse()
