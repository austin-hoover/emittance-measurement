"""Reconstruct the covariance matrix from measurement data."""
from pprint import pprint
from math import sqrt
from Jama import Matrix

import sys
from least_squares import lsq_linear


def load_array(filename):
    """Load text as 2D list."""
    file = open(filename, 'r')
    M = []
    for line in file:
        line = line.rstrip()
        M.append([float(token) for token in line.split()])
    file.close()
    return M


def to_mat(moment_vec):
    """Return covariance matrix from 10 element moment vector."""
    (sig_11, sig_22, sig_12,
     sig_33, sig_44, sig_34, 
     sig_13, sig_23, sig_14, sig_24) = moment_vec
    return [[sig_11, sig_12, sig_13, sig_14], 
            [sig_12, sig_22, sig_23, sig_24], 
            [sig_13, sig_23, sig_33, sig_34], 
            [sig_14, sig_24, sig_34, sig_44]]


def to_vec(Sigma):
    """Return 10 element moment vector from covariance matrix."""
    s11, s12, s13, s14 = Sigma[0][:]
    s22, s23, s24 = Sigma[1][1:]
    s33, s34 = Sigma[2][2:]
    s44 = Sigma[3][3]
    return np.array([s11, s22, s12, s33, s44, s34, s13, s23, s14, s24])


def reconstruct(transfer_mats, moments, **lsq_kws):
    """Reconstruct covariance matrix from measured moments and transfer matrices.
    
    Parameters
    ----------
    transfer_mats : list
        Each element is a list of shape (4, 4) representing a transfer matrix.
    moments : list
        Each element is list containing of [cov(x, x), cov(y, y), cov(x, y)], 
        where cov means covariance.
    **lsq_kws
        Key word arguments passed to `lsq_linear` method.
        
    Returns
    -------
    list, shape (4, 4)
        Reconstructed covariance matrix.
    """
    # Form A and b
    A, b = [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(transfer_mats, moments):
        A.append([M[0][0]**2, M[0][1]**2, 2*M[0][0]*M[0][1], 0, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, M[2][2]**2, M[2][3]**2, 2*M[2][2]*M[2][3], 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, M[0][0]*M[2][2],  M[0][1]*M[2][2],  M[0][0]*M[2][3],  M[0][1]*M[2][3]])
        b.append(sig_xx)
        b.append(sig_yy)
        b.append(sig_xy)
        
    # Solve the problem.
    lsq_kws.setdefault('solver', 'lsmr')
    moment_vec = lsq_linear(A, b, **lsq_kws)
    Sigma = Matrix(to_mat(moment_vec))
    return Sigma

#     print 'eps_x = {} [mm mrad]'.format(eps_x)
#     print 'eps_y = {} [mm mrad]'.format(eps_y)
#     print 'eps_1 = {} [mm mrad]'.format(eps_1)
#     print 'eps_2 = {} [mm mrad]'.format(eps_2)
#     print 'eps_4D = {} [mm^2 mrad^2]'.format(eps_1 * eps_2)