"""
This script sets the phase advance at one wire-scanner in the RTBT using the 
online model. It scans a range of phases, each time writing two files. 

Important variables
-------------------
ref_ws_id : str
    ID of the wirescanner at which the phase advance will be measured. Options: 
    {'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24'}
phase_coverage : float
    The horizontal and vertical phases are varied by this many radians during
    the scan. For example, suppose phase_coverage = T and the default phase
    advances are mux and muy. Then the horizontal phase is varied in the range
    (mux - T/2, mux + T/2) and the vertical phase is varied in the range 
    (muy - T/2, muy + T/2). Ideally this is equal to pi radians.
nsteps_per_dim : int
    Number of phases to measure in each dimension. 
    `x_phases = np.linspace(mux_min, mux_max, nsteps_per_dim)`,
    `y_phases = np.linspace(muy_min, muy_max, nsteps_per_dim)`.
    
Output files
------------
Let w = [WS02, WS20, WS21, WS23, WS24] be a list of the RTBT wire-scanners.
Also let i be the scan index, i.e., the ith phase advance measured in the scan. 
Then the following files are produced for each i:
* 'transfer_matrix_elems_i.dat': 
     The jth row in the file gives the 16 transfer matrix elements from s = 0 
     to wire-scanner w[j]. The elements are written in the order: [M11, M12,
     M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44].
* 'moments_i.dat': 
     The jth row in the file gives [Sigma_11, Sigma_33, Sigma_13] at 
     wire-scanner w[j], where Sigma_mn is the m,n entry in the transverse beam
     covariance matrix with m and n running from 1 to 4. 
* 'model_fields_i.dat':
    ID and field strength of every independent quadrupole.
"""
from lib.phase_controller import PhaseController, ws_ids
from lib.phase_controller import init_twiss, design_betas_at_target
from lib.helpers import loadRTBT, write_traj_to_file
from lib.utils import radians, multiply, delete_files_not_folders


# Setup
#------------------------------------------------------------------------------
delete_files_not_folders('./output/')

# Create lattice and phase controller
sequence = loadRTBT()
ref_ws_id = 'RTBT_Diag:WS24' 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Settings
phase_coverage = radians(180)
nsteps_per_dim = 6
beta_lims = (40, 40)
max_beta = 100

# Save wire-scanner indices in trajectory (for plotting)
file = open('output/ws_index_in_trajectory.dat', 'w')
for ws_id in ws_ids:
    index = controller.trajectory.indicesForElement(ws_id)[0]
    file.write('name = {}, index = {}\n'.format(ws_id, index))
file.close()


# Scan
#------------------------------------------------------------------------------
phases = controller.get_phases_for_scan(phase_coverage, nsteps_per_dim)
mux0, muy0 = controller.get_ref_ws_phases()

print 'Initial phases at {}: {:.3f}, {:.3f}'.format(ref_ws_id, mux0, muy0)
print 'Phase coverage = {:.3f} rad'.format(phase_coverage)
print 'Scan | mux  | muy [rad]'
print '--------------------------'
for i, (mux, muy) in enumerate(phases, start=1):
    print '{:<4} | {:.2f} | {:.2f}'.format(i, mux, muy)

for scan_index, (mux, muy) in enumerate(phases, start=1):
    
    print 'Scan {}/{}'.format(scan_index, 2 * nsteps_per_dim)
    print 'Setting phases at {}.'.format(ref_ws_id)
    controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
    print 'Setting betas at target.'
    controller.set_betas_at_target(design_betas_at_target, max_beta, verbose=1)
    print '  Max betas anywhere: {:.3f}, {:.3f}'.format(*controller.get_max_betas(stop=None))
    
    # Save Twiss vs. position data
    filename = 'output/twiss_{}.dat'.format(scan_index)
    write_traj_to_file(controller.get_twiss(), controller.positions, filename)

    # Save transfer matrix at each wire-scanner
    file = open('output/transfer_mat_elems_{}.dat'.format(scan_index),'w')
    fstr = 16 * '{} ' + '\n'
    for ws_id in ws_ids:
        M = controller.get_transfer_matrix_at(ws_id)
        elements = [elem for row in M for elem in row]
        file.write(fstr.format(*elements))
    file.close()

    # Save real space beam moments at each wire-scanner
    file = open('output/moments_{}.dat'.format(scan_index), 'w')
    for ws_id in ws_ids:
        moments = controller.get_moments_at(ws_id)
        file.write('{} {} {}\n'.format(*moments))
    file.close()
    
    # Save model quadrupole strengths
    file = open('output/model_fields_{}.dat'.format(scan_index), 'w')
    for quad_id in controller.ind_quad_ids:
        field = controller.get_field(quad_id)
        file.write('{}, {}\n'.format(quad_id, field))
    file.close()
    
    print ''
    
# Save phases at each scan index
file = open('output/phases.dat', 'w')
for (mux, muy) in phases:
    file.write('{}, {}\n'.format(mux, muy))
file.close()

exit()