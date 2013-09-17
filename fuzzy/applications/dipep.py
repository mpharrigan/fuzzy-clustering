"""Apply fuzzy clustering to alanine dipeptide from msmbuilder tutorial."""

from fuzzy import get_data
import os
from mdtraj.geometry import dihedral
import numpy as np
from fuzzy import mixture, analysis, classic, buildmsm
import sklearn.mixture
from matplotlib import pyplot as pp

class Dipeptide():

    INDICES = [
               [6, 8, 14, 16],  # Psi
               [4, 6, 8, 14]]  # Phi

    def __init__(self, directory="/home/harrigan/projects/msmbuilder/Tutorial/"):

        # Load from disk
        print "Loading trajectories"
        traj_list_in = get_data.get_from_fs(os.path.join(directory, 'XTC'), os.path.join(directory, 'native.pdb'))

        # Compute dihedrals
        traj_list_angles = list()
        traj_list = list()

        print "Computing Dihedrals"
        for traj in traj_list_in:
            # Compute actual angles
            diheds = dihedral.compute_dihedrals(traj, Dipeptide.INDICES, opt=False)


            # Compute sin and cos
            num_diheds = diheds.shape[1]
            per_diheds = np.zeros((diheds.shape[0], num_diheds * 2))
            per_diheds[:, 0:num_diheds] = np.sin(diheds)
            per_diheds[:, num_diheds:num_diheds * 2] = np.cos(diheds)

            # Save
            traj_list.append(per_diheds)
            traj_list_angles.append(diheds)

        self.traj_list = traj_list
        self.traj_list_angles = traj_list_angles

    def compute_hmm_from_mm(self, t_matrix=None, lag_time=1):
        points = get_data.get_points_from_trajlist(self.traj_list)
        _, memberships = self.mixture_model.score_samples(points)

        # Build an initial MSM as a guess
        print("Building guess transition matrix")
        rev_counts, t_matrix, populations, mapping = \
                    buildmsm.build_classic_from_memberships(memberships, lag_time=lag_time)


        # Learn from trajectories in HMM
        print("Performing Baum-Welch algorithm")
        hidden_mm = mixture.get_hidden_markov_model(self.mixture_model, t_matrix)
        hidden_mm = mixture.perform_optimization(hidden_mm, self.traj_list, lag_time=lag_time)

        # Get the transition matrix in the normal form
        new_t_matrix = hidden_mm.asMatrices()[0]
        self.hmm = hidden_mm
        self.redo_mm()
        print "Done."
        return new_t_matrix, hidden_mm

    def compute_hmm(self, k):
        lag_time = 1
        mm_stride = 10
        sliding_window = True
        if k is not None:
            t_matrix, new_t_matrix, hidden_mm, first_mixture_model, opt_mixture_model = mixture.hmm(self.traj_list, fix_k=k, lag_time=lag_time, mm_stride=mm_stride, sliding_window=sliding_window)
        else:
            t_matrix, new_t_matrix, hidden_mm, first_mixture_model, opt_mixture_model = mixture.hmm(self.traj_list, min_k=15, max_k=25, lag_time=lag_time, mm_stride=mm_stride, sliding_window=sliding_window)

        self.old_t_matrix = t_matrix
        self.new_t_matrix = new_t_matrix
        self.hmm = hidden_mm
        self.first_mixture_model = first_mixture_model
        self.opt_mixture_model = opt_mixture_model

        print "Done."

    def redo_mm(self):
        matrices = self.hmm.asMatrices()
        means = np.array([matrices[1][j][0] for j in xrange(len(matrices[1]))])
        n_components = means.shape[0]
        n_features = means.shape[1]
        covars = np.array([matrices[1][j][1] for j in xrange(len(matrices[1]))])
        covars = np.reshape(covars, (n_components, n_features, n_features))
        weights = np.array(matrices[2])

        mm = sklearn.mixture.GMM(n_components=n_components, params='', init_params='', covariance_type='full')
        mm.means_ = means
        mm.covars_ = covars
        mm.weights_ = weights


        self.new_mixture_model = mm

    def plot(self, new=True):
        if new:
            mm = self.opt_mixture_model
        else:
            mm = self.first_mixture_model

        means = mm.means_
        meansconv = np.zeros((means.shape[0], 2))
        meansconv[:, 0] = _from_sincos(means[:, ::2])
        meansconv[:, 1] = _from_sincos(means[:, 1::2])
        meansconv *= (180. / np.pi)

#         meansconv = self.mixture_model.means_ * (180. / np.pi)

        for i in xrange(len(self.traj_list)):

            traj_act = self.traj_list_angles[i] * (180. / np.pi)
            traj_per = self.traj_list[i]

            _, memberships = mm.score_samples(traj_per)
            analysis.plot_points_with_alpha(traj_act[:, [1, 0]], memberships)

        # Plot means
        pp.scatter(meansconv[:, 1], meansconv[:, 0], facecolors='w', edgecolors='k', s=100, zorder=10)

    def plot_classic(self, ass, n_clusters):
        means = self.mixture_model.means_
        meansconv = np.zeros((means.shape[0], 2))
        meansconv[:, 0] = _from_sincos(means[:, ::2])
        meansconv[:, 1] = _from_sincos(means[:, 1::2])
        meansconv *= (180. / np.pi)

#         meansconv = self.mixture_model.means_ * (180. / np.pi)

        for i in xrange(len(self.traj_list)):

            traj_act = self.traj_list_angles[i] * (180. / np.pi)

            memberships = list()
            for j in xrange(len(traj_act)):
                mem = np.zeros(n_clusters)
                mem[ass[i, j]] = 1.0
                memberships.append(mem)
            memberships = np.array(memberships)
            analysis.plot_points_with_alpha(traj_act[:, [1, 0]], memberships)


        # Plot means
        pp.scatter(meansconv[:, 1], meansconv[:, 0], facecolors='w', edgecolors='k', s=100, zorder=10)

    def medoids(self, n_clusters, dim):
        hkm = classic.cluster(self.traj_list, n_clusters, n_medoid_iters=0, dim=dim)

        if not hasattr(self, 'mixture_model'):
            self.mixture_model = sklearn.mixture.GMM(n_components=dim, params='wmc', init_params='', covariance_type='full')

        self.mixture_model.means_ = hkm.get_generators_as_traj()['XYZList'][:, 0, :]
        self.mixture_model.covars_ = np.array([np.identity(dim) * 0.1 for _ in xrange(n_clusters)])
        self.mixture_model.weights_ = np.ones(n_clusters)
        self.mixture_model.n_components = n_clusters
        return hkm


def _from_sincos(sincos):
    return np.arctan2(sincos[:, 0], sincos[:, 1])

