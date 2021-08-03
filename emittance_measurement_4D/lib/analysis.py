"""Reconstruct covariance matrix from measurement data."""
from __future__ import print_function
import sys
import math
from math import sqrt
from pprint import pprint
from datetime import datetime
from Jama import Matrix

from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager

# Local
from least_squares import lsq_linear
from optics import TransferMatrixGenerator
import utils
import xal_helpers


DIAG_WIRE_ANGLE = utils.radians(-45.0)


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


def twiss2D(Sigma):
    eps_x, eps_y = apparent_emittances(Sigma)
    beta_x = Sigma.get(0, 0) / eps_x
    beta_y = Sigma.get(2, 2) / eps_y
    alpha_x = -Sigma.get(0, 1) / eps_x
    alpha_y = -Sigma.get(2, 3) / eps_y
    return [alpha_x, alpha_y, beta_x, beta_y]


class BeamStats:
    """Container for beam statistics calculated from the covariance matrix."""
    def __init__(self, Sigma):
        if type(Sigma) is list:
            Sigma = Matrix(Sigma)
        self.Sigma = Sigma
        self.eps_x, self.eps_y = apparent_emittances(Sigma)
        self.eps_1, self.eps_2 = intrinsic_emittances(Sigma)
        self.alpha_x, self.alpha_y, self.beta_x, self.beta_y = twiss2D(Sigma)
        self.coupling_coeff = 1.0 - sqrt(self.eps_1 * self.eps_2 / (self.eps_x * self.eps_y))
        
    def rms_ellipse_dims(dim1, dim2):
        return rms_ellipse_dims(self.Sigma, dim1, dim2)
    
    def print_all(self):
        print('eps_1, eps_2 = {} {} [mm mrad]'.format(self.eps_1, self.eps_2))
        print('eps_x, eps_y = {} {} [mm mrad]'.format(self.eps_x, self.eps_y))
        print('alpha_x, alpha_y = {} {} [rad]'.format(self.alpha_x, self.alpha_y))
        print('beta_x, beta_y = {} {} [m/rad]'.format(self.beta_x, self.beta_y))


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
    # Form A and b.
    A, b = [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(transfer_mats, moments):
        A.append([M[0][0]**2, M[0][1]**2, 2*M[0][0]*M[0][1], 0, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, M[2][2]**2, M[2][3]**2, 2*M[2][2]*M[2][3], 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, M[0][0]*M[2][2],  M[0][1]*M[2][2],  M[0][0]*M[2][3],  M[0][1]*M[2][3]])
        b.append(sig_xx)
        b.append(sig_yy)
        b.append(sig_xy)

    # Solve the problem Ax = b.
    lsq_kws.setdefault('solver', 'lsmr')
    moment_vec = lsq_linear(A, b, **lsq_kws)
    Sigma = to_mat(moment_vec)
    Sigma = Matrix(Sigma)
    return Sigma



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

        # Read the lines
        profiles = dict()
        self.node_ids = sorted(list(lines))
        for node_id in sorted(list(self.node_ids)):
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

            profile = Profile(
                [xpos, ypos, upos], 
                [xraw, yraw, uraw], 
                [xfit, yfit, ufit], 
                [xstats, ystats, ustats],
            )
            self[node_id] = profile
            
    def get_moments(self):
        """Store/return dictionary of measured moments at each profile."""
        self.moments = dict()
        for node_id in self.node_ids:
            profile = self[node_id]
            sig_xx = profile.hor.stats['Sigma'].rms**2
            sig_yy = profile.ver.stats['Sigma'].rms**2
            sig_uu = profile.dia.stats['Sigma'].rms**2
            sig_xy = get_sig_xy(sig_xx, sig_yy, sig_uu, profile.diag_wire_angle)
            self.moments[node_id] = [sig_xx, sig_yy, sig_xy]
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
        """Write files in nice format (nicer than PTA files at least)."""
        return
    
    
def get_scan_info(measurements, tmat_generator, start_node_id):
    """Make dictionaries of measured moments and transfer matrices at each wire-scanner."""
    moments_dict, tmats_dict = dict(), dict()
    for measurement in measurements:
        filename = measurement.filename.split('/')[-1]
        print("Reading file '{}'  pvloggerid = {}".format(filename, measurement.pvloggerid))
        measurement.get_moments()
        for node_id, moments in measurement.moments.items():
            if node_id not in moments_dict:
                moments_dict[node_id] = []
            moments_dict[node_id].append(moments)
        measurement.get_transfer_mats(start_node_id, tmat_generator)
        for node_id, tmat in measurement.transfer_mats.items():
            if node_id not in tmats_dict:
                tmats_dict[node_id] = []
            tmats_dict[node_id].append(tmat)
    print('All files have been read.')
    return moments_dict, tmats_dict