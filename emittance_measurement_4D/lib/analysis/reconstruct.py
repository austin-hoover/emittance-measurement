"""Reconstruct the covariance matrix from measurement data."""
from pprint import pprint
from math import sqrt
from Jama import Matrix

import sys
from .least_squares import lsq_linear
from .. import utils


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


# Read PTA files
#-------------------------------------------------------------------------------
def string_to_float_list(string):
    """Convert string to list of floats.
    
    '1 2 3' -> [1.0, 2.0, 3.0])
    """
    return [float(token) for token in string.split()]



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
        Each key is a different statistical parameter: ('Area' or 'Mean' or ...). 
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
    """
    def __init__(self, pos, raw, fit=None, stats=None):
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
        
        
class Measurement:
    """Holds data for one measurement.
    
    Each measurement is a collection of wire scans at a single machine setting.
    """
    def __init__(self, filename):
        self.profiles = dict()
        self.pvloggerid = None
        self.filename = filename
        self.read_pta_file()
        
    def read_pta_file(self):
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

        # Read the lines
        profiles = dict()
        for ws_id in sorted(list(lines)):
            # Split lines into three sections:
            #     stats: statistical signal parameters;
            #     raw: wire positions and raw signal amplitudes;
            #     fit: wire positions and Gaussian fit amplitudes.
            # There is one blank line after each section.
            sep = ''
            lines_stats, lines_raw, lines_fit = utils.split_list(lines[ws_id], sep)[:3]

            # Remove headers and dashed lines beneath headers.
            lines_stats = lines_stats[2:]
            lines_raw = lines_raw[2:]
            lines_fit = lines_fit[2:]   

            # The columns of the following array are ['pos', 'yraw', 'uraw', 'xraw', 
            # 'xpos', 'ypos', 'upos']. (NOTE: This is not the order that is written
            # in the file header.)
            data_arr_raw = [string_to_float_list(line) for line in lines_raw]
            pos, yraw, uraw, xraw, xpos, ypos, upos = utils.transpose_list(data_arr_raw)

            # This next array is the same, but it contains 'yfit', 'ufit', 'xfit', 
            # instead of 'yraw', 'uraw', 'xraw'.
            data_arr_fit = [string_to_float_list(line) for line in lines_fit]
            pos, yfit, ufit, xfit, xpos, ypos, upos = utils.transpose_list(data_arr_fit)

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

            self.profiles[ws_id] = Profile(
                [xpos, ypos, upos], 
                [xraw, yraw, uraw], 
                [xfit, yfit, ufit], 
                [xstats, ystats, ustats],
            )