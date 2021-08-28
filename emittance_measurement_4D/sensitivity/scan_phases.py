"""
Scan the horizontal and vertical phase advances at WS24. 

At each (mux, muy) in the scan:
    1. Calculate the sum of the condition numbers of the coefficient matrices Axx, Ayy, Axy.
    2. Run Monte Carlo simulation and recored fail rate, mean emittances, and standard deviation
       of emittances.
"""
from __future__ import print_function
import sys
import os
import math
import random
from Jama import Matrix

from xal.ca import Channel
from xal.ca import ChannelFactory
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager
from xal.smf.impl import MagnetMainSupply
from xal.tools.beam import CovarianceMatrix
from xal.tools.beam import PhaseVector
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings

from helpers import get_moments
from helpers import run_trials
from helpers import solve
from helpers import uncoupled_matched_cov

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import analysis
from lib import optics
from lib.least_squares import lsq_linear 
from lib import utils


utils.delete_files_not_folders('_output/')


# Setup
ws_ids = ['RTBT_Diag:WS02', 'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
n_steps = 10
dmux_lo = utils.radians(-45.0)
dmux_hi = utils.radians(+45.0)
dmuy_lo = utils.radians(-45.0)
dmuy_hi = utils.radians(+45.0)
controller = optics.PhaseController(ref_ws_id=ws_ids[-1], sync_mode='live')
mux0, muy0 = controller.phases(ws_ids[-1])

# Create initial matched covariance matrix (uncoupled).
eps_x = 32.0 # [mm mrad]
eps_y = 25.0 # [mm mrad]
alpha_x = controller.init_twiss['alpha_x']
alpha_y = controller.init_twiss['alpha_y']
beta_x = controller.init_twiss['beta_x']
beta_y = controller.init_twiss['beta_y']
Sigma0 = uncoupled_matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y)




n_trials = 1000
rms_frac_size_error = 0.03


# Compute condition number for Axy over grid of x and y phase advances.
#------------------------------------------------------------------------------
dmuxx = utils.linspace(dmux_lo, dmux_hi, n_steps)
dmuyy = utils.linspace(dmuy_lo, dmuy_hi, n_steps)
condition_numbers = Matrix(n_steps, n_steps)
fail_rates = Matrix(n_steps, n_steps)
eps_x_means = Matrix(n_steps, n_steps)
eps_y_means = Matrix(n_steps, n_steps)
eps_1_means = Matrix(n_steps, n_steps)
eps_2_means = Matrix(n_steps, n_steps)
eps_x_stds = Matrix(n_steps, n_steps)
eps_y_stds = Matrix(n_steps, n_steps)
eps_1_stds = Matrix(n_steps, n_steps)
eps_2_stds = Matrix(n_steps, n_steps)

def cond(A):
    if A.getRowDimension() == A.getColumnDimension():
        Ainv = A.inverse()
    else:
        AT = A.transpose()
        Ainv = ((AT.times(A)).inverse()).times(AT)
    return A.normF() * Ainv.normF()
        

print()
print('i/N  | j/N  | mux   | muy   | cond(Axy)  ')
print('-----------------------------------------')
for i, dmuy in enumerate(dmuyy):    
    muy = utils.put_angle_in_range(muy0 + dmuy)
    for j, dmux in enumerate(dmuxx):
        mux = utils.put_angle_in_range(mux0 + dmux)
        controller.set_ref_ws_phases(mux, muy, verbose=0)
        tmats = [controller.transfer_matrix(ws_id) for ws_id in ws_ids]
        Axx, Ayy, Axy = [], [], []
        for M in tmats:
            Axx.append([M[0][0]**2, M[0][1]**2, 2*M[0][0]*M[0][1]])
            Ayy.append([M[2][2]**2, M[2][3]**2, 2*M[2][2]*M[2][3]])
            Axy.append([M[0][0]*M[2][2],  M[0][1]*M[2][2],  M[0][0]*M[2][3],  M[0][1]*M[2][3]])

        Axx = Matrix(Axx)
        Ayy = Matrix(Ayy)
        Axy = Matrix(Axy)
        condition_number = cond(Axx) + cond(Ayy) + cond(Axy)
        
        tmats = [controller.transfer_matrix(ws_id) for ws_id in ws_ids]
        fail_rate, emittances = run_trials(Sigma0, tmats, n_trials, rms_frac_size_error)
        means = utils.mean_cols(emittances)
        stds = utils.std_cols(emittances)
        eps_x_means.set(i, j, means[0])
        eps_y_means.set(i, j, means[1])
        eps_1_means.set(i, j, means[2])
        eps_2_means.set(i, j, means[3])
        eps_x_stds.set(i, j, stds[0])
        eps_y_stds.set(i, j, stds[1])
        eps_1_stds.set(i, j, stds[2])
        eps_2_stds.set(i, j, stds[3])       
        
        fail_rates.set(i, j, fail_rate)
        condition_numbers.set(i, j, condition_number)
        
        
        fstr = '{}/{} | {}/{} | {:.3f} | {:.3f} | {} | {} | {} | {} {}'
        print(fstr.format(i, n_steps - 1, j, n_steps - 1, mux, muy, condition_number,
                          means[0], means[1], means[2], means[3]))
        
utils.save_array(condition_numbers.getArray(), '_output/condition_numbers.dat')
utils.save_array(fail_rates.getArray(), '_output/fail_rates.dat')
utils.save_array(eps_x_means.getArray(), '_output/eps_x_means.dat')
utils.save_array(eps_y_means.getArray(), '_output/eps_y_means.dat')
utils.save_array(eps_1_means.getArray(), '_output/eps_1_means.dat')
utils.save_array(eps_2_means.getArray(), '_output/eps_2_means.dat')
utils.save_array(eps_x_stds.getArray(), '_output/eps_x_stds.dat')
utils.save_array(eps_y_stds.getArray(), '_output/eps_y_stds.dat')
utils.save_array(eps_1_stds.getArray(), '_output/eps_1_stds.dat')
utils.save_array(eps_2_stds.getArray(), '_output/eps_2_stds.dat')
utils.save_array(dmuxx, '_output/phase_devs_x.dat')
utils.save_array(dmuyy, '_output/phase_devs_y.dat')

exit()