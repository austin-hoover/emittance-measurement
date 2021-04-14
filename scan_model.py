"""
This script performs the phase scan using the linear model. For each scan
it writes the files 'transfer_matrix_elems_i.dat' and 'moments_i.dat', where i
is the scan number. Each of row in these files corresponds to a different 
wire-scanner.
"""
from lib.phase_controller import PhaseController, all_quad_ids, ws_ids
from lib.phase_controller import init_twiss, design_betas_at_target
from lib.helpers import loadRTBT, write_traj_to_file
from lib.utils import radians, multiply, delete_files_not_folders


# Setup
#------------------------------------------------------------------------------
delete_files_not_folders('./output/')

sequence = loadRTBT()

# Create phase controller
ref_ws_id = 'RTBT_Diag:WS24' # scan phases at this wire-scanner
init_twiss['ex'] = init_twiss['ey'] = 20e-6 # arbitrary [m*rad] 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Settings
phase_coverage = radians(180)
scans_per_dim = 5
beta_lims = (40, 40)
beta_lim_after_ws24 = 100

# Save wire-scanner indices in trajectory (for plotting)
file = open('output/ws_index_in_trajectory.dat', 'w')
for ws_id in ws_ids:
    index = controller.trajectory.indicesForElement(ws_id)[0]
    file.write('name = {}, index = {}\n'.format(ws_id, index))
file.close()


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

for scan_index, (mux, muy) in enumerate(phases, start=1):
    
    print 'Scan {}/{}'.format(scan_index, 2 * scans_per_dim)
    print 'Setting phases at {}.'.format(ref_ws_id)
    controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
    print 'Setting betas at target.'
    controller.set_betas_at_target(design_betas_at_target, beta_lim_after_ws24, verbose=1)
    print '  Max betas anywhere: {:.3f}, {:.3f}'.format(*controller.get_max_betas(stop_id=None))
    
    # Check if field settings are outside trip limits or warning limits
    print 'Checking field limits.'
    for quad_id in all_quad_ids:
        B = controller.get_field_strength(quad_id, 'model')
        quad_node = sequence.getNodeWithId(quad_id)
        print '  ' + quad_id
        print '    desired field = {:.3f}'.format(B)
        print '    default field = {:.3f}'.format(quad_node.getField())
        lower_field_limit = quad_node.lowerFieldLimit()
        upper_field_limit = quad_node.upperFieldLimit()
        lower_alarm_field_limit = quad_node.lowerAlarmFieldLimit()
        upper_alarm_field_limit = quad_node.upperAlarmFieldLimit()
        if B <= lower_field_limit or B >= upper_field_limit:
            print '    Desired field exceeds field limits!'
        if B <= lower_alarm_field_limit or B >= upper_alarm_field_limit:
            print '    Desired field exceeds alarm field limits!'
        print '    Field limits = {:.3f}, {:.3f}'.format(lower_field_limit, upper_field_limit)
        print '    Alarm field limits = {:.3f}, {:.3f}'.format(lower_alarm_field_limit, upper_alarm_field_limit)

    # Save Twiss vs. position data
    filename = 'output/twiss_{}.dat'.format(scan_index)
    write_traj_to_file(controller.get_twiss(), controller.positions, filename)

    # Save transfer matrix at each wire-scanner. There will be one row per 
    # wire-scanner in the order [ws02, ws20, ws21, ws23, ws24]. Each row lists
    # the 16 elements of the transfer matrix in the order [00, 01, 02, 03, 10,
    # 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33].
    file = open('output/transfer_mat_elems_{}.dat'.format(scan_index),'w')
    fstr = 16 * '{} ' + '\n'
    for ws_id in ws_ids:
        M = controller.get_transfer_matrix_at(ws_id)
        elements = [elem for row in M for elem in row]
        file.write(fstr.format(*elements))
    file.close()

    # Save real space beam moments at each wire-scanner. There will be one row 
    # per wire-scanner in the order [ws02, ws20, ws21, ws23, ws24]. Each row 
    # lists [<xx>, <yy>, <xy>].
    file = open('output/moments_{}.dat'.format(scan_index), 'w')
    for ws_id in ws_ids:
        moments = controller.get_moments_at(ws_id)
        file.write('{} {} {}\n'.format(*moments))
    file.close()
    
    # Save model quadrupole strengths.
    file = open('output/quad_settings_{}.dat'.format(scan_index), 'w')
    for quad_id in all_quad_ids:
        field_strength = controller.get_field_strength(quad_id)
        file.write('{}, {}\n'.format(quad_id, field_strength))
    file.close()
    
    print ''
    
# Save phases at each scan index
file = open('output/phases.dat', 'w')
for (mux, muy) in phases:
    file.write('{}, {}\n'.format(mux, muy))
file.close()

exit()