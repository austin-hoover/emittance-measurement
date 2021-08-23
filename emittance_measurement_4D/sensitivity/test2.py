"""
Scan the horizontal and vertical phase advances at WS24. 

At each step in the scan:
    1. Calculate condition number of the coefficient matrices Axx, Ayy, Axy.
    2. Run Monte Carlo simulation.
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


# Setup
ws_ids = optics.RTBT_WS_IDS[1:]
n_steps = 10
dmux_lo = utils.radians(-30.0)
dmux_hi = utils.radians(+45.0)
dmuy_lo = utils.radians(-30.0)
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


# Compute condition number for Axy over grid of x and y phase advances.
#------------------------------------------------------------------------------
dmuxx = utils.linspace(dmux_lo, dmux_hi, n_steps)
dmuyy = utils.linspace(dmuy_lo, dmuy_hi, n_steps)
condition_numbers = Matrix(n_steps, n_steps)
mean_eps_x = Matrix(n_steps, n_steps)
mean_eps_y = Matrix(n_steps, n_steps)
mean_eps_1 = Matrix(n_steps, n_steps)
mean_eps_2 = Matrix(n_steps, n_steps)
std_eps_x = Matrix(n_steps, n_steps)
std_eps_y = Matrix(n_steps, n_steps)
std_eps_1 = Matrix(n_steps, n_steps)
std_eps_2 = Matrix(n_steps, n_steps)

print()
print('i/N  | j/N  | mux   | muy   | cond(Axy)')
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
        
        condition_number = utils.cond(Axy)
        condition_numbers.set(i, j, condition_number)
        
        fstr = '{}/{} | {}/{} | {:.3f} | {:.3f} | {}'
        print(fstr.format(i, n_steps - 1, j, n_steps - 1, mux, muy, condition_number))
        
utils.save_array(condition_numbers.getArray(), 'condition_numbers.dat')
utils.save_array(dmuxx, 'phase_devs_x.dat')
utils.save_array(dmuyy, 'phase_devs_y.dat')

exit()