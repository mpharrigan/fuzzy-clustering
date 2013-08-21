from fuzzy import fcm, mixture, get_data

def compare():
    fcm.demonstrate_classic(big_k=200, num_med_iters=10, lag_time=10, show=True)
    trajs = get_data.get_trajs()
    mixture.hmm(trajs, lag_time=10)
    

if __name__ == "__main__":
    compare()
