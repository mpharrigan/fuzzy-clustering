from fuzzy import mixture
import hmm

def _get_t_matrix(hmm):
    """Get the transition matrix in an msmbuilder format from an HMM.
    
    This is a scipy.sparse.csr_matrix.
    """
    pass

def build_sample_hmm():
    """Build an HMM from a constructed transition matrix and artificial
    emission probabilities.
    """
    t_matrix = []  # TODO
    emissions = []  # TODO
    
    hmm = ghmm.GaussianWhatever(t_matrix, emissions, pi)  # TODO
    
    
    return hmm

def unflatten(dim):
    """Take a flattened trajectory/sequence TODO: update docs
    
    and turn it into normal trajectories."""

def sample_from_hmm(hmm, n_trajs, traj_len):
    hmm.getSequence()  # TODO
    unflatten()
    return


def test_hmm():
    hmm = build_sample_hmm()
    t_matrix_constructed = _get_t_matrix(hmm)
    traj_list = sample_from_hmm(hmm, n_trajs, traj_len)
    
    plot_trajs(traj_list)  # TODO: Check if I already have this function
    
    t_matrix_learned = mixture.hmm(traj_list)
