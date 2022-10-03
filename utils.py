import numpy as np

def KL(P,Q):
    """ Epsilon is used avoid dividing by zero and numerical imprecisions """
    epsilon = 1e-6
    P = P+epsilon
    Q = Q+epsilon
    divergence = np.sum(P*np.log((P)/(Q)))
    return divergence
