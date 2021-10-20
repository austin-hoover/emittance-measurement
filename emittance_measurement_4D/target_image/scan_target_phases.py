"""This script scans the phase advances at the target.

The model is first synchronized with the live machine state. Then the phase
advances (mux, muy) are scanned over a grid. The model phase advances and
quad strengths at each step are saved to a file after the script runs. The
model Twiss parameters throughout the RTBT are saved to a separate file at
each step.

IMPORTANT: set the correct kinetic energy!
"""
from __future__ import print_function
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib.utils import radians, degrees
from lib.xal_helpers import write_traj_to_file

# Settings
kinetic_energy = 1.0e9
n_steps = 4
dmux_min = dmuy_min = radians(-50.)
dmux_max = dmuy_max = radians(124.)
beta_max_before_ws24 = 35.0
beta_max_after_ws24 = 95.0
target_beta_frac_tol = 0.16
quad_ids = ['RTBT_Mag:QH18', 'RTBT_Mag:QV19',
            'RTBT_Mag:QH26', 'RTBT_Mag:QV27', 'RTBT_Mag:QH28', 'RTBT_Mag:QV29', 'RTBT_Mag:QH30']

# Create PhaseController and save default machine state.
controller = optics.PhaseController(kinetic_energy=kinetic_energy)
mux0, muy0 = controller.phases('RTBT:Tgt')
default_target_betas = controller.beta_funcs('RTBT:Tgt')
default_fields = controller.get_fields(quad_ids, 'model')
print('Initial (default):')
print('  Phase advances at target = {:.2f}, {:.2f}'.format(mux0, muy0))
print('  Max betas anywhere (< WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(stop='RTBT_Diag:WS24')))
print('  Max betas anywhere (> WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(start='RTBT_Diag:WS24', stop='RTBT:Tgt')))
print('  Betas at target = {:.2f}, {:.2f}'.format(*controller.beta_funcs('RTBT:Tgt')))

# Create grid of phase advances.
muxx = optics.lin_phase_range(mux0 + dmux_min, mux0 + dmux_max, n_steps, endpoint=False)
muyy = optics.lin_phase_range(muy0 + dmuy_min, muy0 + dmuy_max, n_steps, endpoint=False)

# Initialize files.
file_phase_adv = open('_output/data/phase_adv.dat', 'w')
file_fields = open('_output/data/fields.dat', 'w')
file_default_fields = open('_output/data/default_fields.dat', 'w')
for quad_id in quad_ids:
    file_fields.write(quad_id + ' ')
    file_default_fields.write(quad_id + ' ')
file_fields.write('\n')
file_default_fields.write('\n')

# Perform the scan.
start_time = time.time()
counter = 0
for i, mux in enumerate(muxx):
    for j, muy in enumerate(muyy):
        print('i, j, time = {}, {}, {}'.format(i, j, time.time() - start_time))

        controller.set_target_phases(mux, muy, beta_max_before_ws24, beta_max_after_ws24,
                                     default_target_betas, target_beta_frac_tol,
                                     guess=default_fields)

        mux_calc, muy_calc = controller.phases('RTBT:Tgt')
        fields = controller.get_fields(quad_ids, 'model')

        # Print a progress report.
        print('Final:')
        print('  Phase advances at target = {:.2f}, {:.2f} [deg]'.format(degrees(mux_calc), degrees(muy_calc)))
        print('  Expected                 = {:.2f}, {:.2f} [deg]'.format(degrees(mux), degrees(muy)))
        print('  Max betas anywhere (< WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(stop='RTBT_Diag:WS24')))
        print('  Max betas anywhere (> WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(start='RTBT_Diag:WS24', stop='RTBT:Tgt')))
        print('  Betas at target = {:.2f}, {:.2f}'.format(*controller.beta_funcs('RTBT:Tgt')))
        print('Magnet changes (id, new, old, abs(frac_change):')
        for quad_id, field, default_field in zip(quad_ids, fields, default_fields):
            frac_change = abs(field - default_field) / default_field
            print('  {}: {} {} {}'.format(quad_id, field, default_field, frac_change))

        # Save model phase advances, quadrupole fields, and tracked Twiss parameters to a file.
        file_phase_adv.write('{} {}\n'.format(mux_calc, muy_calc))
        for field, default_field in zip(fields, default_fields):
            file_fields.write('{} '.format(field))
            file_default_fields.write('{} '.format(default_field))
        file_fields.write('\n')
        file_default_fields.write('\n')
        write_traj_to_file(controller.tracked_twiss(), controller.positions, 
                           '_output/data/twiss_{}.dat'.format(counter))

        counter += 1


print('Runtime = {}'.format(time.time() - start_time))
file_phase_adv.close()
file_fields.close()
file_default_fields.close()

exit()