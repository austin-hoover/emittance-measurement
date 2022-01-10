"""Reconstruct covariance matrix from measurement data."""
from __future__ import print_function
import sys
import math
import random
from math import sqrt, sin, cos
from pprint import pprint
from datetime import datetime
from Jama import Matrix

from xal.extension.solver import Scorer
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.sim.scenario import Scenario
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq
from xal.smf.data import XMLDataManager

# Local
from least_squares import lsq_linear
import utils
from xal_helpers import minimize
from xal_helpers import get_trial_vals


DIAG_WIRE_ANGLE = utils.radians(-45.0)
INF = 1e20


# Covariance matrix analysis
#-------------------------------------------------------------------------------
def rms_ellipse_dims(Sigma, dim1, dim2):
    """Return tilt angle and semi-axes of rms ellipse.

    Parameters
    ----------
    Sigma : Matrix, shape (4, 4)
        The covariance matrix for [x, x', y, y']. The rms ellipsoid is defined
        by w^T Sigma w, where w = [x, x', y, y']^T.
    dim1, dim2, {'x', 'xp', 'y', 'yp'}
        The horizontal (dim1) and vertical (dim2) dimension. The 4D ellipsoid
        is projected onto this 2D plane.
        
    Returns
    -------
    phi : float
        The tilt angle of the ellipse as measured below the horizontal axis.
        So, a positive tilt angle means a negative correlation.
    c1, c2 : float
        The horizontal and vertical semi-axes, respectively, of the ellipse
        when phi = 0.
    """
    str_to_int = {'x':0, 'xp':1, 'y':2, 'yp':3}
    i = str_to_int[dim1]
    j = str_to_int[dim2]
    sig_ii, sig_jj, sig_ij = Sigma.get(i, i), Sigma.get(j, j), Sigma.get(i, j)
    phi = -0.5 * math.atan2(2 * sig_ij, sig_ii - sig_jj)
    sn, cs = math.sin(phi), math.cos(phi)
    sn2, cs2 = sn**2, cs**2
    c1 = sqrt(abs(sig_ii*cs2 + sig_jj*sn2 - 2*sig_ij*sn*cs))
    c2 = sqrt(abs(sig_ii*sn2 + sig_jj*cs2 + 2*sig_ij*sn*cs))
    return phi, c1, c2


def intrinsic_emittances(Sigma):
    U = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    SU = Sigma.times(U)
    SU2 = SU.times(SU)
    trSU2 = SU2.trace()
    detS = Sigma.det()
    eps_1 = 0.5 * sqrt(-trSU2 + sqrt(trSU2**2 - 16 * detS))
    eps_2 = 0.5 * sqrt(-trSU2 - sqrt(trSU2**2 - 16 * detS)) 
    return eps_1, eps_2


def apparent_emittances(Sigma):
    eps_x = sqrt(Sigma.get(0, 0) * Sigma.get(1, 1) - Sigma.get(0, 1)**2)
    eps_y = sqrt(Sigma.get(2, 2) * Sigma.get(3, 3) - Sigma.get(2, 3)**2)
    return eps_x, eps_y


def emittances(Sigma):
    eps_x, eps_y = apparent_emittances(Sigma)
    eps_1, eps_2 = intrinsic_emittances(Sigma)
    return eps_x, eps_y, eps_1, eps_2


def twiss2D(Sigma):
    eps_x, eps_y = apparent_emittances(Sigma)
    beta_x = Sigma.get(0, 0) / eps_x
    beta_y = Sigma.get(2, 2) / eps_y
    alpha_x = -Sigma.get(0, 1) / eps_x
    alpha_y = -Sigma.get(2, 3) / eps_y
    return alpha_x, alpha_y, beta_x, beta_y


def is_positive_definite(Sigma):
    """Return True if symmetric matrix is positive definite."""
    return all([eigval >= 0 for eigval in Sigma.eig().getRealEigenvalues()])


def is_valid_covariance_matrix(Sigma):
    """Return True if the covariance matrix `Sigma` makes physical sense."""
    if not is_positive_definite(Sigma):
        return False
    if Sigma.det() < 0:
        return False
    eps_x, eps_y, eps_1, eps_2 = emittances(Sigma)
    if (eps_x * eps_y < eps_1 * eps_2):
        return False
    return True


def V_matrix_uncoupled(alpha_x, alpha_y, beta_x, beta_y):
    """4x4 normalization matrix for x-x' and y-y'."""
    V = Matrix([[sqrt(beta_x), 0, 0, 0],
                [-alpha_x/sqrt(beta_x), 1/sqrt(beta_x), 0, 0],
                [0, 0, sqrt(beta_y), 0],
                [0, 0, -alpha_y/sqrt(beta_y), 1/sqrt(beta_y)]])
    return V


class BeamStats:
    """Container for beam statistics calculated from the covariance matrix."""
    def __init__(self, Sigma, Sigmas=None):
        """Constructor

        Sigma : Jama Matrix, shape (4, 4)
            The covariance matrix.
        Sigmas : list[Jama Matrix or list, shape (4, 4)]
            Ensemble of covariance matrices from random trials.
        """
        self.Sigma = Sigma
        self.eps_x, self.eps_y = apparent_emittances(Sigma)
        if is_valid_covariance_matrix(Sigma):
            self.eps_1, self.eps_2 = intrinsic_emittances(Sigma)
            self.coupling_coeff = sqrt(self.eps_x * self.eps_y / (self.eps_1 * self.eps_2))
        else:
            self.eps_1 = self.eps_2 = self.coupling_coeff = None
        self.alpha_x, self.alpha_y, self.beta_x, self.beta_y = twiss2D(Sigma)

        self.Sigmas = None
        self.ran_eps_x_mean = self.ran_eps_x_std = None
        self.ran_eps_y_mean = self.ran_eps_y_std = None
        self.ran_eps_1_mean = self.ran_eps_1_std = None
        self.ran_eps_2_mean = self.ran_eps_2_std = None
        self.ran_eps_x_eps_y_mean = self.ran_eps_x_eps_y_std = None
        self.ran_eps_1_eps_2_mean = self.ran_eps_1_eps_2_std = None
        self.ran_beta_x_mean = self.ran_beta_x_std = None
        self.ran_beta_y_mean = self.ran_beta_y_std = None
        self.ran_alpha_x_mean = self.ran_alpha_x_std = None
        self.ran_alpha_y_mean = self.ran_alpha_y_std = None

        if Sigmas is not None:
            self.Sigmas = Sigmas

            (ran_eps_x_list,
             ran_eps_y_list,
             ran_eps_1_list,
             ran_eps_2_list) = utils.transpose([emittances(Sigma) for Sigma in Sigmas])
            self.ran_eps_x_mean, self.ran_eps_x_std = utils.mean_std(ran_eps_x_list)
            self.ran_eps_y_mean, self.ran_eps_y_std = utils.mean_std(ran_eps_y_list)
            self.ran_eps_1_mean, self.ran_eps_1_std = utils.mean_std(ran_eps_1_list)
            self.ran_eps_2_mean, self.ran_eps_2_std = utils.mean_std(ran_eps_2_list)

            ran_eps_x_eps_y_list = [eps_x * eps_y for eps_x, eps_y in zip(ran_eps_x_list, ran_eps_y_list)]
            ran_eps_1_eps_2_list = [eps_1 * eps_2 for eps_1, eps_2 in zip(ran_eps_1_list, ran_eps_2_list)]
            self.ran_eps_x_eps_y_mean, self.ran_eps_x_eps_y_std = utils.mean_std(ran_eps_x_eps_y_list)
            self.ran_eps_1_eps_2_mean, self.ran_eps_1_eps_2_std = utils.mean_std(ran_eps_1_eps_2_list)

            (ran_alpha_x_list,
             ran_alpha_y_list,
             ran_beta_x_list,
             ran_beta_y_list) = utils.transpose([twiss2D(Sigma) for Sigma in Sigmas])
            self.ran_beta_x_mean, self.ran_beta_x_std = utils.mean_std(ran_beta_x_list)
            self.ran_beta_y_mean, self.ran_beta_y_std = utils.mean_std(ran_beta_y_list)
            self.ran_alpha_x_mean, self.ran_alpha_x_std = utils.mean_std(ran_alpha_x_list)
            self.ran_alpha_y_mean, self.ran_alpha_y_std = utils.mean_std(ran_alpha_y_list)

    def rms_ellipse_dims(dim1, dim2):
        return rms_ellipse_dims(self.Sigma, dim1, dim2)
    
    def print_all(self):
        print('eps_1, eps_2 = {} {} [mm mrad]'.format(self.eps_1, self.eps_2))
        print('eps_x, eps_y = {} {} [mm mrad]'.format(self.eps_x, self.eps_y))
        print('alpha_x, alpha_y = {} {} [rad]'.format(self.alpha_x, self.alpha_y))
        print('beta_x, beta_y = {} {} [m/rad]'.format(self.beta_x, self.beta_y))


def to_mat(moments):
    """Return covariance matrix from 10 element moment vector."""
    (sig_11, sig_22, sig_12,
     sig_33, sig_44, sig_34, 
     sig_13, sig_23, sig_14, sig_24) = moments
    return Matrix([[sig_11, sig_12, sig_13, sig_14], 
                   [sig_12, sig_22, sig_23, sig_24], 
                   [sig_13, sig_23, sig_33, sig_34], 
                   [sig_14, sig_24, sig_34, sig_44]])


def to_vec(Sigma):
    """Return 10 element moment vector from covariance matrix."""
    return utils.list_to_col_mat(
        [Sigma.get(0, 0), Sigma.get(1, 1), Sigma.get(0, 1),
         Sigma.get(2, 2), Sigma.get(3, 3), Sigma.get(2, 3),
         Sigma.get(0, 2), Sigma.get(1, 2), Sigma.get(0, 3), Sigma.get(1, 3)])

 

# Covariance matrix reconstruction
#-------------------------------------------------------------------------------
def get_sig_xy(sig_xx, sig_yy, sig_uu, diag_wire_angle):
    """Compute cov(x, y) from horizontal, vertical, and diagonal wires.
    
    Diagonal wire angle should be in radians.
    """
    phi = utils.radians(90.0) + diag_wire_angle
    sin, cos = math.sin(phi), math.cos(phi)
    sig_xy = (sig_uu - sig_xx*(cos**2) - sig_yy*(sin**2)) / (2 * sin * cos)
    return sig_xy


def reconstruct(transfer_mats, moments, constr=True, **lsq_kws):
    """Reconstruct covariance matrix from measured moments and transfer matrices.

    Parameters
    ----------
    transfer_mats : list[list, shape (4, 4)], shape (n,)
        Each element is a list of shape (4, 4) representing a transfer matrix
        from the reconstruction location to the measurement location.
    moments : list[list, shape(3,)], shape (n,)
        The measured [<xx>, <yy>, <xy>] moments.
    constr: bool
        Whether to try nonlinear solver if LLSQ answer is unphysical.
        (Default: True)
    **lsq_kws
        Key word arguments passed to `lsq_linear` method.
        
    Returns
    -------
    Jama Matrix, shape (4, 4)
        Reconstructed covariance matrix.
    """
    A, b = [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(transfer_mats, moments):
        A.append([M[0][0]**2, M[0][1]**2, 2*M[0][0]*M[0][1], 0, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, M[2][2]**2, M[2][3]**2, 2*M[2][2]*M[2][3], 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, M[0][0]*M[2][2],  M[0][1]*M[2][2],  M[0][0]*M[2][3],  M[0][1]*M[2][3]])
        b.append(sig_xx)
        b.append(sig_yy)
        b.append(sig_xy)

    lsq_kws.setdefault('solver', 'exact')
    Sigma = to_mat(lsq_linear(A, b, **lsq_kws))
    
    if not constr:
        return Sigma

    if is_valid_covariance_matrix(Sigma):
        print('Covariance matrix is physical.')
        return Sigma
    
    print('Covariance matrix is unphysical. Running solver.')
    A = Matrix(A)
    b = utils.list_to_col_mat(b)
    
    # This section uses the parameterization of Edwards/Teng.
    #
    # We parameterize Sigma as Sigma = V * C * Sigma_n * C^T * V^T. V is simply
    # the block-diagonal normalization matrix in terms of the 2D Twiss 
    # parameters. Sigma_n = diag(eps_1, eps_1, eps_2, eps_2). C is a symplectic
    # matrix with three free parameters: a, b, c. When a = b = c = 0, we have
    # eps_x = eps_1, eps_y = eps_2 (no coupling). When they are nonzero, there
    # is coupling.
    #---------------------------------------------------------------------------
    def get_cov(eps_1, eps_2, alpha_x, alpha_y, beta_x, beta_y, a, b, c):
        E = utils.diagonal_matrix([eps_1, eps_1, eps_2, eps_2])
        V = Matrix(4, 4, 0.)
        V.set(0, 0, sqrt(beta_x))
        V.set(1, 0, -alpha_x / sqrt(beta_x))
        V.set(1, 1, 1. / sqrt(beta_x))
        V.set(2, 2, sqrt(beta_y))
        V.set(3, 2, -alpha_y / sqrt(beta_y))
        V.set(3, 3, 1. / sqrt(beta_y))
        if a == 0:
            if b == 0 or c == 0:
                d = 0
            else:
                raise ValueError("a is zero but b * c is not zero.")
        else:
            d = b * c / a
        C = Matrix([[1, 0, a, b], [0, 1, c, d], [-d, b, 1, 0], [c, -a, 0, 1]])
        VC = V.times(C)
        return VC.times(E.times(VC.transpose()))

    class MyScorer(Scorer):
        
        def __init__(self):
            return
        
        def score(self, trial, variables):
            Sigma = get_cov(*get_trial_vals(trial, variables))
            vec = to_vec(Sigma)
            residuals = A.times(vec).minus(b)
            return residuals.normF()**2
        
    eps_x, eps_y = apparent_emittances(Sigma)  
    alpha_x, alpha_y, beta_x, beta_y = twiss2D(Sigma)
    guess = [eps_x, eps_y, alpha_x, alpha_y, beta_x, beta_y,
             0.1 * random.random(), -0.1 * random.random(), +0.1 * random.random()]
    lb = [0., 0., -INF, -INF, 0., 0., -INF, -INF, -INF]
    ub = INF
    bounds = (lb, ub)
    var_names = ['eps_1', 'eps_2', 'alpha_x', 'alpha_y', 'beta_x', 'beta_y', 'a', 'b', 'c']
    scorer = MyScorer()
    
    params = minimize(scorer, guess, var_names, bounds, maxiters=50000, tol=1e-15, verbose=2)
    Sigma = get_cov(*params)
    return Sigma

#     # This section uses inverse Cholesky factorization Sigma = L * L^T, where
#     # L is a lower triangular matrix.
#     #---------------------------------------------------------------------------
#     def get_cov(x):
#         L = Matrix([[x[0], 0, 0, 0],
#                     [x[1], x[2], 0, 0],
#                     [x[3], x[4], x[5], 0],
#                     [x[6], x[7], x[8], x[9]]])
#         return L.times(L.transpose())
        
#     class MyScorer(Scorer):
        
#         def __init__(self):
#             return
        
#         def score(self, trial, variables):
#             Sigma = get_cov(get_trial_vals(trial, variables))
#             vec = to_vec(Sigma)
#             residuals = A.times(vec).minus(b)
#             return residuals.normF()**2
                
#     bounds = (-INF, INF)
#     guess = [random.random() - 0.5 for _ in range(10)]
#     var_names = ['v_{}'.format(i) for i in range(10)]
#     scorer = MyScorer()
    
#     x = minimize(scorer, guess, var_names, bounds, maxiters=50000, tol=1e-15, verbose=2)
#     Sigma = get_cov(x)
#     return Sigma


def reconstruct_random_trials(transfer_mats, moments, frac_err=0.02, n_trials=1000):
    """Reconstruct with errors added to the measured moments.

    Here `moments` should be a list of [<xx>, <yy>, <uu>].
    """
    Sigmas = []
    fails = 0
    hi = 1.0 + frac_err
    lo = 1.0 - frac_err

    def add_noise(value, frac_err):
        max_err = frac_err * value
        return value + random.uniform(-max_err, +max_err)

    def run_trial(moments, transfer_mats):
        noisy_moments = []
        for i in range(len(moments)):
            sig_xx, sig_yy, sig_uu = moments[i]
            sig_xx = add_noise(sig_xx, frac_err)
            sig_yy = add_noise(sig_yy, frac_err)
            sig_uu = add_noise(sig_uu, frac_err)
            sig_xy = get_sig_xy(sig_xx, sig_yy, sig_uu, DIAG_WIRE_ANGLE)
            noisy_moments.append([sig_xx, sig_yy, sig_xy])
        Sigma = reconstruct(transfer_mats, noisy_moments, constr=False)
        return Sigma

    for _ in range(n_trials):
        failed = True
        n_attempts, max_attempts = 0, 1000
        Sigma = None
        while failed and n_attempts < max_attempts:
            Sigma = run_trial(moments, transfer_mats)
            failed = not is_valid_covariance_matrix(Sigma)
            fails += int(failed)
            n_attempts += 1
        if Sigma is not None:
            Sigmas.append(Sigma)
    return Sigmas



# PTA file processing
#-------------------------------------------------------------------------------
def is_harp_file(filename):
    file = open(filename)
    for line in file:
        if 'Harp' in line:
            return True
    return False


class Stat:
    """Container for a signal parameter.
    
    Attributes
    ----------
    name : str
        Parameter name.
    rms : float
        Parameter value from rms calculation.
    fit : float
        Parameter value from Gaussian fit.
    """
    def __init__(self, name, rms, fit):
        self.name, self.rms, self.fit = name, rms, fit

        
class Signal:
    """Container for profile signal.
    
    Attributes
    ----------
    pos : list
        Wire positions.
    raw : list
        Raw signal amplitudes at each position.
    fit : list
        Gaussian fit amplitudes at each position.
    stats : dict
        Each key is a different statistical parameter: ('Area', 'Mean', etc.). 
        Each value is a Stat object that holds the parameter name, rms value, 
        and Gaussian fit value.
    """
    def __init__(self, pos, raw, fit, stats):
        self.pos, self.raw, self.fit, self.stats = pos, raw, fit, stats
        
        
class Profile:
    """Stores data from single wire-scanner.
    
    Attributes
    ----------
    hor, ver, dia : Signal
        Signal object for horizontal, vertical and diagonal wire.
    diag_wire_angle : float
        Angle of diagonal wire above the x axis.
    """
    def __init__(self, pos, raw, fit=None, stats=None, diag_wire_angle=DIAG_WIRE_ANGLE):
        """Constructor.
        
        Parameters
        ----------
        pos : [xpos, ypos, upos]
            Position lists for each wire.
        raw : [xraw, yraw, uraw]
            List of raw signal amplitudes for each wire.
        fit : [xfit, yfit, ufit]
            List of Gaussian fit amplitudes for each wire.
        stats : [xstats, ystats, ustats]
            List of stats dictionaries for each wire.
        """
        self.diag_wire_angle = diag_wire_angle
        xpos, ypos, upos = pos
        xraw, yraw, uraw = raw
        if fit is None:
            xfit = yfit = ufit = None
        else:
            xfit, yfit, ufit = fit
        if stats is None:
            xstats = ystats = ustats = None
        else:
            xstats, ystats, ustats = stats
        self.hor = Signal(xpos, xraw, xfit, xstats)
        self.ver = Signal(ypos, yraw, yfit, ystats)
        self.dia = Signal(upos, uraw, ufit, ustats)
        

class Measurement(dict):
    """Dictionary of profiles for one measurement.

    Each key in this dictionary is a wire-scanner ID; each value is a Profile.
    
    Attributes
    ----------
    filename : str
        Full path to the PTA file.
    filename_short : str
        Only include the filename, not the full path.
    timestamp : datetime
        Represents the time at which the data was taken.
    pvloggerid : int
        The PVLoggerID of the measurement (this gives a snapshot of the machine state).
    node_ids : list[str]
        The ID of each wire-scanner. (These are the dictionary keys.)
    moments : dict
        The [<x^2>, <y^2>, <xy>] moments at each wire-scanner.
    transfer_mats : dict
        The linear 4x4 transfer matrix from a start node to each wire-scanner. 
        The start node is determined in the function call `get_transfer_mats`.
    """
    def __init__(self, filename):
        dict.__init__(self)
        self.filename = filename
        self.filename_short = filename.split('/')[-1]
        self.timestamp = None
        self.pvloggerid = None
        self.node_ids = None
        self.moments, self.transfer_mats = dict(), dict()
        self.read_pta_file()
        
    def read_pta_file(self):
        # Store the timestamp on the file.
        date, time = self.filename.split('WireAnalysisFmt-')[-1].split('_')
        time = time.split('.pta')[0]
        year, month, day = [int(token) for token in date.split('.')]
        hour, minute, second = [int(token) for token in time.split('.')]
        self.timestamp = datetime(year, month, day, hour, minute, second)
        
        # Collect lines corresponding to each wire-scanner
        file = open(self.filename, 'r')
        lines = dict()
        ws_id = None
        for line in file:
            line = line.rstrip()
            if line.startswith('RTBT_Diag'):
                ws_id = line
                continue
            if ws_id is not None:
                lines.setdefault(ws_id, []).append(line)
            if line.startswith('PVLoggerID'):
                self.pvloggerid = int(line.split('=')[1])
        file.close()
        self.node_ids = sorted(list(lines))

        # Read the lines
        for node_id in self.node_ids:
            # Split lines into three sections:
            #     stats: statistical signal parameters;
            #     raw: wire positions and raw signal amplitudes;
            #     fit: wire positions and Gaussian fit amplitudes.
            # There is one blank line after each section.
            sep = ''
            lines_stats, lines_raw, lines_fit = utils.split(lines[node_id], sep)[:3]

            # Remove headers and dashed lines beneath headers.
            lines_stats = lines_stats[2:]
            lines_raw = lines_raw[2:]
            lines_fit = lines_fit[2:]   

            # The columns of the following array are ['pos', 'yraw', 'uraw', 'xraw', 
            # 'xpos', 'ypos', 'upos']. (NOTE: This is not the order that is written
            # in the file header.)
            data_arr_raw = [utils.string_to_list(line) for line in lines_raw]
            pos, yraw, uraw, xraw, xpos, ypos, upos = utils.transpose(data_arr_raw)

            # This next array is the same, but it contains 'yfit', 'ufit', 'xfit', 
            # instead of 'yraw', 'uraw', 'xraw'.
            data_arr_fit = [utils.string_to_list(line) for line in lines_fit]
            pos, yfit, ufit, xfit, xpos, ypos, upos = utils.transpose(data_arr_fit)

            # Get statistical signal parameters. (Headers don't give true ordering.)
            xstats, ystats, ustats = dict(), dict(), dict()
            for line in lines_stats:
                tokens = line.split()
                name = tokens[0]
                vals = [float(val) for val in tokens[1:]]
                s_yfit, s_yrms, s_ufit, s_urms, s_xfit, s_xrms = vals
                xstats[name] = Stat(name, s_xrms, s_xfit)
                ystats[name] = Stat(name, s_yrms, s_yfit)
                ustats[name] = Stat(name, s_urms, s_ufit)

            self[node_id] = Profile([xpos, ypos, upos], 
                                    [xraw, yraw, uraw],
                                    [xfit, yfit, ufit], 
                                    [xstats, ystats, ustats])
            
    def read_harp_file(self, filename):
        file = open(filename, 'r')
        data = []
        pvloggerid = None
        for line in file:
            tokens = line.rstrip().split()
            if not tokens or tokens[0] in ['start', 'RTBT_Diag:Harp30']:
                continue
            if tokens[0] == 'PVLoggerID':
                pvloggerid = int(tokens[-1])
                continue
            data.append([float(token) for token in tokens])
        file.close()
                
        xpos, xraw, ypos, yraw, upos, uraw = utils.transpose(data)
        self['RTBT_Diag:Harp30'] = Profile([xpos, ypos, upos], [xraw, yraw, uraw])
            
    def get_moments(self):
        """Store/return dict of measured [<xx>, <yy>, <uu>, <xy>] at each profile."""
        self.moments = dict()
        for node_id in self.node_ids:
            profile = self[node_id]
            sig_xx = profile.hor.stats['Sigma'].rms**2
            sig_yy = profile.ver.stats['Sigma'].rms**2
            sig_uu = profile.dia.stats['Sigma'].rms**2
            sig_xy = get_sig_xy(sig_xx, sig_yy, sig_uu, profile.diag_wire_angle)
            self.moments[node_id] = [sig_xx, sig_yy, sig_uu, sig_xy]
        return self.moments

    def get_transfer_mats(self, start_node_id, tmat_generator):
        """Store/return dictionary of transfer matrices from start_node to each profile."""
        self.transfer_mats = dict()
        tmat_generator.sync(self.pvloggerid)
        for node_id in self.node_ids:
            tmat = tmat_generator.generate(start_node_id, node_id)
            self.transfer_mats[node_id] = tmat
        return self.transfer_mats

    def export_files(self):
        raise NotImplementedError


def process(filenames):
    if type(filenames) is not list:
        filenames = [filenames]
    filenames = [filename for filename in filenames if 'WireAnalysisFmt' in filename]
    ws_filenames = [filename for filename in filenames if not is_harp_file(filename)]
    harp_filenames = [filename for filename in filenames if is_harp_file(filename)]

    # Read all wire-scanner files.
    measurements = [Measurement(filename) for filename in ws_filenames]

    # Read each harp file into the Measurement with the same PVLoggerID.(It seems that the 
    # Harp files will not have the same PVLoggerID as the wire-scanner files, even if they 
    # were taken immediately after. So this section needs to change.
    harp_pvloggerid = dict()
    for filename in harp_filenames:
        file = open(filename, 'r')
        for line in file:
            pass
        pvloggerid = int(line.split()[-1])
        harp_pvloggerid[pvloggerid] = filename
    for measurement in measurements:
        if measurement.pvloggerid in harp_pvloggerid:
            filename = harp_pvloggerid[measurement.pvloggerid]
            measurement.read_harp_file(filename)

    measurements = sorted(measurements, key=lambda measurement: measurement.timestamp)
    measurements = [measurement for measurement in measurements 
                    if measurement.pvloggerid > 0
                    and measurement.pvloggerid is not None]
    return measurements


class DictOfLists(dict):
    def __init__(self):
        dict.__init__(self)
        
    def add(self, key, value):
        if key not in self:
            self[key] = []
        self[key].append(value)
    

def get_scan_info(measurements, tmat_generator, start_node_id):
    """Make dictionaries of measured moments and transfer matrices at each wire-scanner."""
    print('Reading files...')
    if type(measurements) is not list:
        measurements = [measurements]
    moments_dict, tmats_dict = DictOfLists(), DictOfLists()
    for measurement in measurements:
        print("  Reading file '{}'  pvloggerid = {}".format(measurement.filename_short, 
                                                            measurement.pvloggerid))
        measurement.get_moments()
        measurement.get_transfer_mats(start_node_id, tmat_generator)
        for node_id in measurement.node_ids:
            moments_dict.add(node_id, measurement.moments[node_id])
            tmats_dict.add(node_id, measurement.transfer_mats[node_id])
    print('Done.')
    return moments_dict, tmats_dict