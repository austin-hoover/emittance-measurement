"""Study the effect of mismatch on the fixed-optics reconstruction."""
from __future__ import print_function
import sys
import os
import math
import pickle
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


# Setup
kinetic_energy = 1.0e9 # [eV]
ws_ids = ['RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
ref_ws_id = 'RTBT_Diag:WS24'
n_trials = 100
frac_error = 0.02
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
rec_node_id = 'RTBT_Diag:BPM17'
_, _, alpha_x0, alpha_y0, beta_x0, beta_y0, _, _ = controller.twiss(rec_node_id)

# Move to new optics setting.
mux = utils.put_angle_in_range(mux0 + utils.radians(45.0))
muy = utils.put_angle_in_range(muy0 - utils.radians(45.0))
controller.set_ref_ws_phases(mux, muy, verbose=1)
tmats = [controller.transfer_matrix(rec_node_id, ws_id) for ws_id in ws_ids]

# Study error as function of mismatched beam Twiss parameters.
dalpha_x_min = -0.1
dalpha_x_max = +0.1
dalpha_y_min = -0.1
dalpha_y_max = +0.5
dbeta_x_min = -0.1
dbeta_x_max = +0.1
dbeta_y_min = -0.1
dbeta_y_max = +0.1
n = 10

alpha_x_min = alpha_x0 * (1.0 + dalpha_x_min)
alpha_x_max = alpha_x0 * (1.0 + dalpha_x_max)
alpha_xs = utils.linspace(alpha_x_min, alpha_x_max, n)

alpha_y_min = alpha_y0 * (1.0 + dalpha_y_min)
alpha_y_max = alpha_y0 * (1.0 + dalpha_y_max)
alpha_ys = utils.linspace(alpha_y_min, alpha_y_max, n)

beta_x_min = beta_x0 * (1.0 + dbeta_x_min)
beta_x_max = beta_x0 * (1.0 + dbeta_x_max)
beta_xs = utils.linspace(beta_x_min, beta_x_max, n)

beta_y_min = beta_y0 * (1.0 + dbeta_y_min)
beta_y_max = beta_y0 * (1.0 + dbeta_y_max)
beta_ys = utils.linspace(beta_y_min, beta_y_max, n)

print(alpha_xs)
print(alpha_ys)
print(beta_xs)
print(beta_ys)



def init_array(n):
    array = [[[[[0.0, 0.0, 0.0, 0.0] 
                for i in range(n)] 
               for j in range(n)] 
              for k in range(n)] 
             for l in range(n)]
    return array

emittance_means_list = init_array(n)
emittance_stds_list = init_array(n)
fail_rates = [[[[0.0 for i in range(n)] for j in range(n)] for k in range(n)] for l in range(n)]
for i, alpha_x, in enumerate(alpha_xs):
    for j, alpha_y in enumerate(alpha_ys):
        for k, beta_x in enumerate(beta_xs):
            print(i, j, k)
            for l, beta_y in enumerate(beta_ys):
                Sigma0 = uncoupled_matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_1, eps_2)
                fail_rate, emittances = run_trials(Sigma0, tmats, n_trials, frac_error)
                means = utils.mean_cols(emittances)
                stds = utils.std_cols(emittances)
                for m in range(4):
                    emittance_means_list[i][j][k][l][m] = means[m]
                    emittance_stds_list[i][j][k][l][m] = stds[m]
                fail_rates[i][j][k][l] = fail_rate

def save(array, filename, pkl=False):
    file = open(filename, 'wb' if pkl else 'w')
    if pkl:
        pickle.dump(array, file)
    else:
        for x in array:
            file.write('{} '.format(x))
    file.close()
                    
save(fail_rates, '_output/data/fail_rates.pkl', pkl=True)
save(emittance_means_list, '_output/data/emittance_means.pkl', pkl=True)
save(emittance_stds_list, '_output/data/emittance_stds.pkl', pkl=True)
save(alpha_xs, '_output/data/alpha_xs.dat')
save(alpha_ys, '_output/data/alpha_ys.dat')
save(beta_xs, '_output/data/beta_xs.dat')
save(beta_ys, '_output/data/beta_ys.dat')
save([alpha_x0, alpha_y0, beta_x0, beta_y0], '_output/data/true_twiss.dat')
save([eps_1, eps_2], '_output/data/true_emittances.dat')

exit()