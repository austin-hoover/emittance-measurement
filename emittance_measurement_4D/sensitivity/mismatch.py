"""Study the effect of mismatch on the fixed-optics reconstruction."""
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
from helpers import get_cov

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import analysis
from lib import optics
from lib.least_squares import lsq_linear 
from lib import utils


utils.delete_files_not_folders('_output/')


# Setup
kinetic_energy = 1.0e9 # [eV]
ws_ids = ['RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
ref_ws_id = 'RTBT_Diag:WS24'
n_trials = 1
frac_error = 0.0
controller = optics.PhaseController(ref_ws_id=ref_ws_id, 
                                    kinetic_energy=kinetic_energy)

# Sync with PVLoggerID
pvloggerid = 48842340
# pvloggerid = None
if pvloggerid is not None:
    controller.sync_model_pvloggerid(pvloggerid)
    
print('Betas at target:', controller.beta_funcs('RTBT:Tgt'))
print('Phase advances at target:', controller.phases(ref_ws_id))

mux0, muy0 = controller.phases(ref_ws_id)
quad_ids = controller.ind_quad_ids
default_fields = controller.get_fields(quad_ids)

# Lattice Twiss parameters.
eps_1 = 20.1e-6 # [m rad]
eps_2 = 20.0e-6 # [m rad]
alpha_x0 = controller.init_twiss['alpha_x']
alpha_y0 = controller.init_twiss['alpha_y']
beta_x0 = controller.init_twiss['beta_x']
beta_y0 = controller.init_twiss['beta_y']

# Move to new optics setting.
mux = utils.put_angle_in_range(mux0 + utils.radians(45.0))
muy = utils.put_angle_in_range(muy0 + utils.radians(45.0))
controller.set_ref_ws_phases(mux, muy, verbose=1)
tmats = [controller.transfer_matrix(ws_id) for ws_id in ws_ids]

# Study error as function of mismatched beam Twiss parameters.
dalpha = 0.05
dbeta = 0.3
n = 10

alpha_x_min = alpha_x0 * (1.0 - dalpha)
alpha_x_max = alpha_x0 * (1.0 + dalpha)
alpha_xs = utils.linspace(alpha_x_min, alpha_x_max, n)

alpha_y_min = alpha_y0 * (1.0 - dalpha)
alpha_y_max = alpha_y0 * (1.0 + dalpha)
alpha_ys = utils.linspace(alpha_y_min, alpha_y_max, n)

beta_x_min = beta_x0 * (1.0 - dbeta)
beta_x_max = beta_x0 * (1.0 + dbeta)
beta_xs = utils.linspace(beta_x_min, beta_x_max, n)

beta_y_min = beta_y0 * (1.0 - dbeta)
beta_y_max = beta_y0 * (1.0 + dbeta)
beta_ys = utils.linspace(beta_y_min, beta_y_max, n)


emittances_list = [[[[[0.0, 0.0, 0.0, 0.0] 
                      for i in range(n)] 
                     for j in range(n)] 
                    for k in range(n)]
                   for l in range(n)]

for i, alpha_x, in enumerate(alpha_xs):
    for j, alpha_y in enumerate(alpha_ys):
        for k, beta_x in enumerate(beta_xs):
            for l, beta_y in enumerate(beta_ys):
                print(i, j, k, l)
                Sigma0 = uncoupled_matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_1, eps_2)
                fail_rate, emittances = run_trials(Sigma0, tmats, n_trials, frac_error)
                means = utils.mean_cols(emittances)
                for m in range(4):
                    emittances_list[i][j][k][l][m] = means[m]
                means = utils.mean_cols(emittances)
                stds = utils.std_cols(emittances)

import pickle
output = open('_output/data/data.pkl', 'wb')
pickle.dump(emittances_list, output)
output.close()

def write_array_to_file(array, filename):
    file = open(filename, 'w')
    for x in array:
        file.write('{} '.format(x))
    file.close()
    return
    
write_array_to_file(alpha_xs, '_output/data/alpha_xs.dat')
write_array_to_file(alpha_xs, '_output/data/alpha_ys.dat')
write_array_to_file(beta_xs, '_output/data/beta_xs.dat')
write_array_to_file(beta_ys, '_output/data/beta_ys.dat')

exit()