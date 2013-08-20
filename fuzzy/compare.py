from fuzzy import fcm, mixture

def compare():
    fcm.demonstrate_classic(big_k=200, num_med_iters=1, lag_time=10, show=True)
    mixture.test_mixture(min_k=5, max_k=20)


if __name__ == "__main__":
    compare()
