"""Apply fuzzy clustering to alanine dipeptide from msmbuilder tutorial."""

from fuzzy import get_data
import os
from mdtraj.geometry import dihedral
import numpy as np
from fuzzy import mixture, analysis
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
            for dihed_num in xrange(diheds.shape[1]):
                per_diheds[:, dihed_num * 2] = np.sin(diheds[:, dihed_num])
                per_diheds[:, dihed_num * 2 + 1] = np.cos(diheds[:, dihed_num])

            # Save
            traj_list.append(per_diheds)
            traj_list_angles.append(diheds)

        self.traj_list = traj_list
        self.traj_list_angles = traj_list_angles

    def compute_hmm(self, k):
        if k is not None:
            t_matrix, hmm = mixture.hmm(self.traj_list, fix_k=k, lag_time=1)
        else:
            t_matrix, hmm = mixture.hmm(self.traj_list, lag_time=1, mm_stride=10)
        self.t_matrix = t_matrix
        self.hmm = hmm
        self.redo_mm()
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

        print means

        self.mixture_model = mm

    def plot(self):
        means = self.mixture_model.means_
        meansconv = np.zeros((means.shape[0], 2))
        meansconv[:, 0] = _radian(means[:, 0:2])
        meansconv[:, 1] = _radian(means[:, 2:4])
        meansconv *= (180. / np.pi)

        for i in xrange(len(self.traj_list)):

            traj_act = self.traj_list_angles[i] * (180. / np.pi)
            traj_per = self.traj_list[i]

            plot_stride = 50
            memberships = self.mixture_model.predict_proba(traj_per)
            _, memberships = self.mixture_model.score_samples(traj_per)
            analysis.plot_points_with_alpha(traj_act[::plot_stride, [1, 0]], memberships[::plot_stride, ...])

        pp.scatter(meansconv[:, 1], meansconv[:, 0], color='k', s=100, zorder=10)


def _radian(sincos):
    return np.arctan2(sincos[:, 0], sincos[:, 1])

