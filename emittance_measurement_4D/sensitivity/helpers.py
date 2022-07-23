from __future__ import print_function
import sys
import os
import math
from math import sqrt
import random
from Jama import Matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib import analysis
from lib import utils
from lib.least_squares import lsq_linear


def get_moments(Sigma0, tmats, frac_error=None):
    """Return moments after application of each transfer matrix.
    
    Parameters
    ----------
    Sigma0 : Matrix, shape (4, 4)
        Covariance matrix at reconstruction point.
    tmats : list
        Transfer matrices from reconstruction point to measurement points.
    frac_error : float
        The squared moment along each dimension is multiplied by 1 + f, where
        -frac_err <= f <= +frac_err.
        
    Returns
    -------
    moments : list, shape (len(tmats), 3)
        [<xx>, <yy>, <xy>] at each measurement point.
    """
    moments = []
    for M in tmats:
        if type(M) is list:
            M = Matrix(M)
        Sigma = M.times(Sigma0.times(M.transpose()))
        sig_xx = Sigma.get(0, 0)
        sig_yy = Sigma.get(2, 2)
        sig_xy = Sigma.get(0, 2)
        sig_uu = 0.5 * (2.0 * sig_xy + sig_xx + sig_yy)
        if frac_error is not None:
            sig_xx *= 1.0 + random.uniform(-frac_error, frac_error)
            sig_yy *= 1.0 + random.uniform(-frac_error, frac_error)
            sig_uu *= 1.0 + random.uniform(-frac_error, frac_error)
        sig_xy = 0.5 * (2.0 * sig_uu - sig_xx - sig_yy)
        moments.append([sig_xx, sig_yy, sig_xy])
    return moments


def solve(tmats, moments, return_type='matrix'):
    """Return the LLSQ solution from the measurements."""
    Axx, Ayy, Axy = [], [], []
    bxx, byy, bxy = [], [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(tmats, moments):
        Axx.append([M[0][0] ** 2, M[0][1] ** 2, 2 * M[0][0] * M[0][1]])
        Ayy.append([M[2][2] ** 2, M[2][3] ** 2, 2 * M[2][2] * M[2][3]])
        Axy.append(
            [M[0][0] * M[2][2], M[0][1] * M[2][2], M[0][0] * M[2][3], M[0][1] * M[2][3]]
        )
        bxx.append(sig_xx)
        byy.append(sig_yy)
        bxy.append(sig_xy)
    sig_11, sig_22, sig_12 = lsq_linear(Axx, bxx)
    sig_33, sig_44, sig_34 = lsq_linear(Ayy, byy)
    sig_13, sig_23, sig_14, sig_24 = lsq_linear(Axy, bxy)
    Sigma = [
        [sig_11, sig_12, sig_13, sig_14],
        [sig_12, sig_22, sig_23, sig_24],
        [sig_13, sig_23, sig_33, sig_34],
        [sig_14, sig_24, sig_34, sig_44],
    ]
    if return_type == 'matrix':
        Sigma = Matrix(Sigma)
    return Sigma


def run_trials(Sigma0, tmats, n_trials, frac_error=None, disp=False):
    """Repeat measurement `n_trials` times.
    
    Returns
    -------
    fail_rate : float
        Fraction of failed trials.
    emittances : list, shape (n_trials - n_fail, 4)
        Reconstructed [eps_x, eps_y, eps_1, eps_2] at each successful trial.
    """
    emittances, n_fail = [], 0
    for i in range(n_trials):
        moments = get_moments(Sigma0, tmats, frac_error)
        Sigma = solve(tmats, moments)
        if not analysis.is_valid_covariance_matrix(Sigma):
            n_fail += 1
            if disp:
                print(i, "Failed.")
            continue
        eps_x, eps_y, eps_1, eps_2 = analysis.emittances(Sigma)
        emittances.append([eps_x, eps_y, eps_1, eps_2])
        if disp:
            print(i, eps_x, eps_y, eps_1, eps_2)
    fail_rate = float(n_fail) / n_trials
    if not emittances:
        emittances = [[0.0, 0.0, 0.0, 0.0]]
    return fail_rate, emittances


def run_trials2(Sigma0, tmats, n_trials, frac_error=None, disp=False):
    """Repeat measurement `n_trials` times.
    
    Returns
    -------
    fail_rate : float
        Fraction of failed trials.
    Sigmas : list, shape (n_trials - n_fail, 4, 4)
        Reconstructed [eps_x, eps_y, eps_1, eps_2] at each successful trial.
    """
    Sigmas, n_fail = [], 0
    for i in range(n_trials):
        moments = get_moments(Sigma0, tmats, frac_error)
        Sigma = solve(tmats, moments, return_type='list')
        if not analysis.is_valid_covariance_matrix(Matrix(Sigma)):
            n_fail += 1
            if disp:
                print(i, "Failed.")
            continue
        Sigmas.append(Sigma)
        if disp:
            print(i, Sigma)
    if len(Sigmas) == 0:
        Sigmas = 4 * [4 * [0.0]]
    fail_rate = float(n_fail) / n_trials
    return fail_rate, Sigmas


def uncoupled_matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y):
    """Return uncoupled covariance matrix from 2D Twiss parameters."""
    gamma_x = (1 + alpha_x ** 2) / beta_x
    gamma_y = (1 + alpha_y ** 2) / beta_y
    Sigma = Matrix(4, 4, 0.0)
    Sigma.set(0, 0, eps_x * beta_x)
    Sigma.set(1, 1, eps_x * gamma_x)
    Sigma.set(0, 1, -eps_x * alpha_x)
    Sigma.set(2, 2, eps_y * beta_y)
    Sigma.set(3, 3, eps_y * gamma_y)
    Sigma.set(2, 3, -eps_y * alpha_y)
    for i in range(4):
        for j in range(i):
            Sigma.set(i, j, Sigma.get(j, i))
    return Sigma


def matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y, c=0.0):
    """Return uncoupled covariance matrix from 2D Twiss parameters.

    `c` is a number in the range [0.0, 1.0] that represents "how much
    cross-plane correlation is in the beam". If c == 0.0, there is no
    cross-plane correlation. If c == 1.0, we have a Danilov distribution.
    """
    # Start in normalize phase space (normalized in x-x' and y-y').
    Sigma = Matrix(4, 4, 0.0)
    for i in range(4):
        Sigma.set(i, i, 1.0)
    # Add cross-plane correlation.
    Sigma.set(0, 3, +c)
    Sigma.set(3, 0, +c)
    Sigma.set(1, 2, -c)
    Sigma.set(2, 1, -c)
    # Unnormalize x-x' and y-y'.
    V = Matrix(
        [
            [math.sqrt(beta_x), 0.0, 0.0, 0.0],
            [-alpha_x / math.sqrt(beta_x), 1.0 / math.sqrt(beta_x), 0.0, 0.0],
            [0.0, 0.0, math.sqrt(beta_y), 0.0],
            [0.0, 0.0, -alpha_y / math.sqrt(beta_y), 1.0 / math.sqrt(beta_y)],
        ]
    )
    A = Matrix(
        [
            [math.sqrt(eps_x), 0.0, 0.0, 0.0],
            [0.0, math.sqrt(eps_x), 0.0, 0.0],
            [0.0, 0.0, math.sqrt(eps_y), 0.0],
            [0.0, 0.0, 0.0, math.sqrt(eps_y)],
        ]
    )
    VA = V.times(A)
    Sigma = VA.times(Sigma.times(VA.transpose()))
    return Sigma
