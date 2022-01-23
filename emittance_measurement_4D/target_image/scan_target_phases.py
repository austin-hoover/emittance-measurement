"""This script scans the phase advances at the target.

The model is first synchronized with the live machine state. Then the phase
advances (mux, muy) are scanned over a grid. Files are saved containing the
model Twiss parameters, transfer matrices, and quadrupole fields. The 
quadrupole fields are read by the script 'set_target_phases.py'. 

IMPORTANT: set the correct kinetic energy!
"""
from __future__ import print_function
import math
import random
import sys
import os
import time
from xal.service.pvlogger import RemoteLoggingCenter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib import utils
from lib.utils import degrees
from lib.utils import radians
from lib.xal_helpers import write_traj_to_file


# Settings
kinetic_energy = 0.8e9
steps_per_dim = 12
dmux_min = dmuy_min = radians(-50.0)
dmux_max = dmuy_max = radians(124.0)
beta_max_before_ws24 = 30.0
beta_max_after_ws24 = 95.0 
target_beta_frac_tol = 0.16 
quad_ids = [
    'RTBT_Mag:QH18', 
    'RTBT_Mag:QV19', 
    'RTBT_Mag:QH26', 
    'RTBT_Mag:QV27',
    'RTBT_Mag:QH28', 
    'RTBT_Mag:QV29', 
    'RTBT_Mag:QH30'
]

# Create PhaseController and save default machine state.
controller = optics.PhaseController(kinetic_energy=kinetic_energy)
mux0, muy0 = controller.phases('RTBT:Tgt')
default_target_betas = controller.beta_funcs('RTBT:Tgt')
default_fields = controller.get_fields(quad_ids, 'model')

file = open('_output/data/default_twiss.dat', 'w')
file.write('node_id mu_x mu_y alpha_x alpha_y beta_x beta_y\n')
for node in controller.sequence.getNodes():
    mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y, _, _ = controller.twiss(node.getId())
    file.write('{} {} {} {} {} {} {}\n'
               .format(node.getId(), mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y))
file.close()

def print_progress_report(controller):
    print('Phase advances = {:.2f}, {:.2f}'.format(*controller.phases('RTBT:Tgt')))
    print('Max betas (BPM17 - WS24) = {:.2f}, {:.2f}'
          .format(*controller.max_betas(start='RTBT_Diag:BPM17', stop='RTBT_Diag:WS24')))
    print('Max betas (WS24 - Target) =  {:.2f}, {:.2f}'
          .format(*controller.max_betas(start='RTBT_Diag:WS24', stop='RTBT:Tgt')))
    print('Betas at target = {:.2f}, {:.2f}'.format(*controller.beta_funcs('RTBT:Tgt')))

print('Initial (default) settings:')
print_progress_report(controller)

# Create grid of phase advances.
muxx = optics.lin_phase_range(mux0 + dmux_min, mux0 + dmux_max, steps_per_dim, endpoint=False)
muyy = optics.lin_phase_range(muy0 + dmuy_min, muy0 + dmuy_max, steps_per_dim, endpoint=False)

# Save the expected phase advances.
file = open('_output/data/phase_adv_exp.dat', 'w')
for mux in muxx:
    for muy in muyy:
        file.write('{} {}\n'.format(mux, muy))
file.close()
file1 = open('_output/data/muxx.dat', 'w')
file2 = open('_output/data/muyy.dat', 'w')
for mux, muy in zip(muxx, muyy):
    file1.write('{} '.format(mux))
    file2.write('{} '.format(muy))
file1.close()
file2.close()

# Open files.
file_phase_adv = open('_output/data/phase_adv.dat', 'w')
file_fields = open('_output/data/fields.dat', 'w')
for quad_id in quad_ids:
    file_fields.write(quad_id + ' ')
file_fields.write('\n')

# Perform the scan.
start_time = time.time()
step = 0
for i, mux in enumerate(muxx):
    for j, muy in enumerate(muyy):
        print('i, j = {}, {}'.format(i, j))
        
        # Set the model phase advances at the target.
        controller.set_target_phases(mux, muy, 
                                     beta_max_before_ws24, beta_max_after_ws24,
                                     default_target_betas, target_beta_frac_tol,
                                     guess=default_fields)
        
        # Try again if it didn't work.
        mux_calc, muy_calc = controller.phases('RTBT:Tgt')
        cost = math.sqrt((mux - mux_calc)**2 + (muy - muy_calc)**2)
        attempt = 1
        while cost > utils.radians(0.1) and attempt < 7:
            print('Trying again... attempt {}'.format(attempt))
            df = 0.01
            los = utils.multiply(default_fields, 1.0 - df)
            his = utils.multiply(default_fields, 1.0 + df)
            guess = [random.uniform(lo, hi) for lo, hi in zip(los, his)]
            controller.set_target_phases(mux, muy, 
                                         beta_max_before_ws24, beta_max_after_ws24,
                                         default_target_betas, target_beta_frac_tol,
                                         guess=guess)
            mux_calc, muy_calc = controller.phases('RTBT:Tgt')
            cost = math.sqrt((mux - mux_calc)**2 + (muy - muy_calc)**2)
            attempt += 1
            
        # Print a progress report.
        print('Phase advances (expected) = {:.2f}, {:.2f}'.format(mux, muy))
        print_progress_report(controller)                
        print('Magnet changes (id, new, old, abs_diff, abs_fac_diff:')
        fields = controller.get_fields(quad_ids, 'model')
        for quad_id, field, default_field in zip(quad_ids, fields, default_fields):
            abs_diff = abs(field - default_field)
            abs_frac_diff = abs_diff / default_field
            print('  {}: {:.3f} {:.3f} {:.3f} {:.3f}'
                  .format(quad_id, field, default_field, abs_diff, abs_frac_diff))
            
        # Save quadrupole fields, phase advances, and Twiss parameters.
        for field in fields:
            file_fields.write('{} '.format(field))
        file_fields.write('\n')
        file_phase_adv.write('{} {}\n'.format(*controller.phases('RTBT:Tgt')))
        write_traj_to_file(controller.tracked_twiss(), controller.positions, 
                           '_output/data/twiss_{}.dat'.format(step))
        
        # Save transfer matrices from each node to the target.
        file = open('_output/data/tmats_{}.dat'.format(step), 'w')
        file.write('node_id M11 M12 M13 M14 M21 M22 M23 M34 M31 M32 M33 M34 M41 M42 M43 M44\n')
        for node in controller.sequence.getNodes():
            M = controller.transfer_matrix(node.getId(), 'RTBT:Tgt')
            tmat_elems = [M[k][l] for k in range(4) for l in range(4)]
            file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'
                       .format(node.getId(), *tmat_elems))
        file.close()
        
#         # Save PV Logger ID. (This is returning -1 currently.)
#         pvloggerid = controller.snapshot()
#         print('PV Logger ID = {}'.format(pvloggerid))

        step += 1
        ellapsed_time = time.time() - start_time
        time_per_step = ellapsed_time / step
        steps_remaining = steps_per_dim**2 - step
        est_time_remaining = steps_remaining * time_per_step 
        print('Ellapsed time = {:.2f} [s]'.format(ellapsed_time))
        print('Estimated remaining time = {:.2f} [s]'.format(est_time_remaining))

print('Runtime = {} [s]'.format((time.time() - start_time)))
file_phase_adv.close()
file_fields.close()
exit()