"""
This script performs the phase scan manually. For each scan
it writes the file 'transfer_matrix_elems_i.dat', where i is the scan number. 
Each of row of the file corresponds to a different wire-scanner.
"""
from lib.phase_controller import PhaseController, all_quad_ids, ws_ids
from lib.phase_controller import init_twiss, design_betas_at_target
from lib.helpers import loadRTBT, write_traj_to_file
from lib.utils import radians, multiply, delete_files_not_folders


# Setup
#------------------------------------------------------------------------------
delete_files_not_folders('./output/')

# Create phase controller
sequence = loadRTBT()
ref_ws_id = 'RTBT_Diag:WS24' # scan phases at this wire-scanner
init_twiss['ex'] = init_twiss['ey'] = 20e-6 # arbitrary [m*rad] 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Settings
phase_coverage = radians(155)
scans_per_dim = 5
beta_lims = (40, 40)
beta_lim_after_ws24 = 100
scan_index = 0


# Scan
#------------------------------------------------------------------------------
phases = controller.get_phases_for_scan(phase_coverage, scans_per_dim)
mux0, muy0 = controller.get_ref_ws_phases()

print 'Initial phases at {}: {:.3f}, {:.3f}'.format(ref_ws_id, mux0, muy0)
print 'Phase coverage = {:.3f} rad'.format(phase_coverage)
print 'Scan | mux  | muy [rad]'
print '--------------------------'
for i, (mux, muy) in enumerate(phases, start=1):
    print '{:<4} | {:.2f} | {:.2f}'.format(i, mux, muy)
print ''

# Set phase advance at reference wire-scanner
print 'Scan index = {}.'.format(scan_index)
print 'Setting phases at {}.'.format(ref_ws_id)
controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
print 'Setting betas at target.'
controller.set_betas_at_target(design_betas_at_target, beta_lim_after_ws24, verbose=1)
controller.sync_live_quads_with_model(all_quad_ids)

# Save transfer matrix at each wire-scanner. There will be one row per 
# wire-scanner in the order [ws02, ws20, ws21, ws23, ws24]. Each row lists
# the 16 elements of the transfer matrix in the order [00, 01, 02, 03, 10,
# 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33].
file = open('output/transfer_matrix_elements_{}.dat'.format(scan_index), 'w')
fstr = 16 * '{} ' + '\n'
for ws_id in ws_ids:
    M = controller.get_transfer_matrix_at(ws_id)
    elements = [elem for row in M for elem in row]
    file.write(fstr.format(*elements))
file.close()

# Save model quadrupole strengths.
file = open('output/quad_settings_{}.dat'.format(scan_index), 'w')
for quad_id in all_quad_ids:
    field_strength_model = controller.get_field_strength(quad_id, 'model')
#     field_strength_live = controller.get_field_strength(quad_id, 'live')
    file.write('{}, {}\n'.format(quad_id, field_strength_model))
file.close()

# Wire-scanner data needs to collected externally using WireScanner app.
# ...

# Save phases at each scan index
file = open('output/phases.dat', 'w')
for (mux, muy) in phases:
    file.write('{}, {}\n'.format(mux, muy))
file.close()