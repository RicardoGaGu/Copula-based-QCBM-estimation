import scipy as sp
import numpy as np
import seaborn as sns
from collections import Counter

def gaussian_copula(n, P):
    '''
    n: Number of samples
    p: target correlation matrix
    The marginals distributions are also gaussian.
    '''
    d = P.shape[1] # num variables

    # Independent Normal distributions
    Z = np.random.normal(loc=0, scale=1, size=d*n).reshape(n,d)
    # Cholesky Decomposition
    cholesky = np.linalg.cholesky(P)
    # Apply Cholesky to add targeted correlation
    Z_corr = np.dot(Z,cholesky)

    # Multivariate normal distribution with a certain covariance
    #z = np.random.multivariate_normal(mean=m.reshape(d,), cov=K, size=n)
    #Z_corr = np.transpose(z)
    # Probability integral transform
    U = sp.stats.norm.cdf(Z_corr)
    # Inverse CDF
    G = sp.stats.norm.ppf(U)

    return cholesky,Z,U,G

def convert_data_to_binary_string(U,m):

    """ m: number of bits to digitize pseudo-samples in copula space
        U: Pseudo-sample space matrix
    """
    # Discretizing the variables bounded between 0 and 1 into 2^m levels, and assign binary string for each bin
    bins = np.linspace(0, 1, num=2**m)
    inds = np.digitize(U, bins)
    bit_data = []
    for ind in inds:
        # Contenate all variables into single bit string
        bit_sample = ""
        for ind_v in ind:
            bit_sample_var = '{:0{size}b}'.format(ind_v, size=m)
            bit_sample += bit_sample_var

        bit_data.append(bit_sample)

    return bit_data

def empirical_distribution(binary_samples, N_qubits):

    ''' This method outputs the empirical probability distribution given samples in a list of binary strings
    as a dictionary, with keys as outcomes, and values as probabilities. It is used as the target distribution
    training the QCBM circuit '''

    counts = Counter(binary_samples)
    for element in counts:
        '''Convert occurences to relative frequencies of binary string'''
        counts[element] = counts[element]/(len(binary_samples))
    # Make sure all binary strings are represented over the space of probabilities
    for index in range(0, 2**N_qubits):
        '''If a binary string has not been seen in samples, set its value to zero'''
        if '{:0{size}b}'.format(index, size=N_qubits)  not in counts:
            counts['{:0{size}b}'.format(index, size=N_qubits)] = 0

    sorted_binary_samples_dict = {}
    keylist = sorted(counts)
    for key in keylist:
        sorted_binary_samples_dict[key] = counts[key]

    return np.asarray(list(sorted_binary_samples_dict.values()))
