"""
Observe the effect of changing the lattice optics on the measurement 
sensitivity.
"""
from __future__ import print_function
import os
import math
import random
import sys
from Jama import Matrix

from helpers import run_trials
from helpers import uncoupled_matched_cov

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib import utils
from lib.least_squares import lsq_linear 


# Setup
n_trials = 10000
rms_frac_size_errors = utils.linspace(0.0, 0.05, 6)
dmux = utils.radians(+45.0)
dmuy = utils.radians(-45.0)
ws_ids = optics.RTBT_WS_IDS[1:]
controller = optics.PhaseController(ref_ws_id=ws_ids[-1], sync_mode='live')
mux0, muy0 = controller.phases('RTBT_Diag:WS24')


# For printing
def print_header():
    header = ''.join(['error | fail rate | ',
                      'epsx [mm mrad] | ',
                      'epsy [mm mrad] | ',
                      'eps1 [mm mrad] | ',
                      'eps2 [mm mrad] '])
    print(header)
    print(len(header) * '-')

def print_results(rms_frac_size_error, fail_rate, emittances):
    fstr = ' | '.join(['{:<5}', '{:<9.3f}'] + 4 * ['{:<5.2f} +/- {:.2f}'])
    means = utils.mean_cols(emittances)
    stds = utils.std_cols(emittances)
    print(fstr.format(rms_frac_size_error, fail_rate, 
                      means[0], stds[0], means[1], stds[1], 
                      means[2], stds[2],means[3], stds[3]))
    

# Create initial matched covariance matrix (uncoupled).
eps_x = 32.0 # [mm mrad]
eps_y = 25.0 # [mm mrad]
alpha_x = controller.init_twiss['alpha_x']
alpha_y = controller.init_twiss['alpha_y']
beta_x = controller.init_twiss['beta_x']
beta_y = controller.init_twiss['beta_y']
Sigma0 = uncoupled_matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y)



# Record the fail rate vs. error in measured beam size.

def run_and_print_results(controller):
    print_header()
    for rms_frac_size_error in rms_frac_size_errors:
        tmats = [controller.transfer_matrix(ws_id) for ws_id in ws_ids]
        fail_rate, emittances = run_trials(Sigma0, tmats, n_trials, 
                                           rms_frac_size_error)
        print_results(rms_frac_size_error, fail_rate, emittances)

print('Initial phases at WS24: mux, muy = {:.3f}, {:.3f} [rad]'.format(mux0, muy0))
run_and_print_results(controller)
print()
print('Changing optics.')
print('dmux, dmuy = {:.3f}, {:.3f} [deg]'.format(utils.degrees(dmux), 
                                                 utils.degrees(dmuy)))
mux = utils.put_angle_in_range(mux0 + dmux)
muy = utils.put_angle_in_range(muy0 + dmuy)
controller.set_ref_ws_phases(mux, muy, verbose=1)
print()
run_and_print_results(controller)

exit()