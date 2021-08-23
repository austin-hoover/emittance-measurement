from __future__ import print_function
import sys
import os
import math
import random
from Jama import Matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import analysis
from lib import utils
from lib.least_squares import lsq_linear 


def get_moments(Sigma0, tmats, rms_frac_size_error=None):
    """Return moments after application of each transfer matrix.
    
    Parameters
    ----------
    Sigma0 : Matrix, shape (4, 4)
        Covariance matrix at reconstruction point.
    tmats : list
        Transfer matrices from reconstruction point to measurement points.
    rms_frac_size_error : float
        The rms beam size along each dimension is multiplied by 1 + f, where f 
        is normally distributed with standard deviation `rms_frac_size_err`.
        
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
        sig_uu = 0.5 * (2 * sig_xy + sig_xx + sig_yy)
        if rms_frac_size_error:
            sig_xx *= (1 + random.gauss(0, rms_frac_size_error))**2
            sig_yy *= (1 + random.gauss(0, rms_frac_size_error))**2
            sig_uu *= (1 + random.gauss(0, rms_frac_size_error))**2
        sig_xy = 0.5 * (2 * sig_uu - sig_xx - sig_yy)
        moments.append([sig_xx, sig_yy, sig_xy])
    return moments


def solve(tmats, moments):
    """Return the LLSQ solution from the measurements."""
    Axx, Ayy, Axy = [], [], []
    bxx, byy, bxy = [], [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(tmats, moments):            
        Axx.append([M[0][0]**2, M[0][1]**2, 2*M[0][0]*M[0][1]])
        Ayy.append([M[2][2]**2, M[2][3]**2, 2*M[2][2]*M[2][3]])
        Axy.append([M[0][0]*M[2][2],  M[0][1]*M[2][2],  M[0][0]*M[2][3],  M[0][1]*M[2][3]])
        bxx.append(sig_xx)
        byy.append(sig_yy)
        bxy.append(sig_xy)
    sig_11, sig_22, sig_12 = lsq_linear(Axx, bxx)
    sig_33, sig_44, sig_34 = lsq_linear(Ayy, byy)
    sig_13, sig_23, sig_14, sig_24 = lsq_linear(Axy, bxy)
    Sigma = analysis.to_mat([sig_11, sig_22, sig_12, 
                             sig_33, sig_44, sig_34, 
                             sig_13, sig_23, sig_14, sig_24])
    return Sigma


def run_trials(Sigma0, tmats, n_trials, rms_frac_size_err=None, disp=False):
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
        moments = get_moments(Sigma0, tmats, rms_frac_size_err)
        Sigma = solve(tmats, moments)
        if not analysis.is_valid_cov(Sigma):
            n_fail += 1
            if disp:
                print(i, 'Failed.')
            continue
        eps_x, eps_y, eps_1, eps_2 = analysis.emittances(Sigma)
        emittances.append([eps_x, eps_y, eps_1, eps_2])
        if disp:
            print(i, eps_x, eps_y, eps_1, eps_2)
    fail_rate = float(n_fail) / n_trials
    if not emittances:
        emittances = [[0., 0., 0., 0.]]
    return fail_rate, emittances


def uncoupled_matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y):
    """Return uncoupled covariance matrix from 2D Twiss parameters."""
    gamma_x = (1 + alpha_x**2) / beta_x
    gamma_y = (1 + alpha_y**2) / beta_y
    Sigma = Matrix(4, 4, 0.)
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