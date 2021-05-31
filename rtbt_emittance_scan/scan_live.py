"""
This script sets the phase advance at one wire-scanner in the RTBT using the 
online model. It then updates the live power supplies to reflect the model. 

Important variables
-------------------
scan_index : int
    Right now the wire-scanners have to be run manually. So we run this script
    once for each index in the scan. This variable determines which index we
    are on.
ref_ws_id : str
    ID of the wirescanner at which the phase advance will be measured. Options: 
    {'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24'}
phase_coverage : float
    The horizontal and vertical phases are varied by this many degrees during
    the scan. For example, suppose phase_coverage = T and the default phase
    advances are mux and muy. Then the horizontal phase is varied in the range
    (mux - T, mux + T) and the vertical phase is varied in the range 
    (muy + T, muy - T). Ideally this is equal to 180 degrees.
npts : int
    Number of phases to measure in each dimension. 
    `x_phases = numpy.linspace(mux_min, mux_max, npts)`,
    `y_phases = numpy.linspace(muy_max, muy_min, npts)`.
    
Output files
------------
Let w = [WS02, WS20, WS21, WS23, WS24] be a list of the RTBT wire-scanners and
let i be the scan index.
* 'transfer_matrix_elems_i.dat': 
     The jth row in the file gives the 16 transfer matrix elements from s = 0 
     to wire-scanner w[j]. The elements are written in the order: [M11, M12,
     M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44].
* 'moments_i.dat': 
     The jth row in the file gives [Sigma_11, Sigma_33, Sigma_13] at 
     wire-scanner w[j], where Sigma_mn is the m,n entry in the transverse beam
     covariance matrix with m and n running from 1 to 4. 
* 'model_fields_i.dat':
    ID and field strength of every independent model quadrupole.
* 'live_fields_i.dat':
    ID and field strength of every independent live quadrupole.
"""
from lib.phase_controller import PhaseController, ws_ids
from lib.phase_controller import init_twiss
from lib.helpers import load_sequence, write_traj_to_file
from lib.utils import radians


# Settings
scan_index = 0
ref_ws_id = 'RTBT_Diag:WS24' 
phase_coverage = 180.0 # deg
npts = 12
beta_lims = (40, 40) # [m]
max_beta_before_target = 100 # [m]
kin_energy = 1.0 # [GeV]
field_set_kws = dict(max_frac_change=0.05, max_iters=100, sleep_time=0.1)


# Setup
#------------------------------------------------------------------------------
sequence = load_sequence('RTBT')
controller = PhaseController(sequence, ref_ws_id, init_twiss, kin_energy)

# Save default Twiss vs. position data.
filename = '_output/twiss_default.dat'
write_traj_to_file(controller.tracked_twiss(), controller.positions, filename)


# Scan
#------------------------------------------------------------------------------
phases = controller.get_phases_for_scan(phase_coverage, npts)
mux0, muy0 = controller.phases(ref_ws_id)
print 'Initial phases at {}: {:.3f}, {:.3f}.'.format(ref_ws_id, mux0, muy0)
print 'Phase coverage = {:.3f} deg.'.format(phase_coverage)
print 'Scan | mux  | muy [rad]'
print '--------------------------'
for i, (mux, muy) in enumerate(phases, start=1):
    print '{:<4} | {:.2f} | {:.2f}'.format(i, mux, muy)

print 'Scan {}/{}.'.format(scan_index, npts)
print 'Setting phases at {}...'.format(ref_ws_id)
mux, muy = phases[scan_index]
controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)

print 'Setting betas at target...'
controller.constrain_size_on_target(max_beta_before_target, verbose=1)
max_betas_anywhere = controller.max_betas(stop=None)
print '  Max betas anywhere: {:.3f}, {:.3f}.'.format(*max_betas_anywhere)

print 'Syncing live quads with model...'
controller.sync_live_with_model(**field_set_kws)

# Save Twiss vs. position data.
filename = '_output/twiss_{}.dat'.format(scan_index)
write_traj_to_file(controller.tracked_twiss(), controller.positions, filename)

# Save transfer matrix at each wire-scanner.
file = open('_output/transfer_mat_elems_{}.dat'.format(scan_index),'w')
fstr = 16 * '{} ' + '\n'
for ws_id in ws_ids:
    M = controller.transfer_matrix(ws_id)
    elements = [elem for row in M for elem in row]
    file.write(fstr.format(*elements))
file.close()

# Save model quadrupole strengths.
file = open('_output/model_fields_{}.dat'.format(scan_index), 'w')
for quad_id in controller.ind_quad_ids:
    field = controller.get_field(quad_id, 'model')
    file.write('{}, {}\n'.format(quad_id, field))
file.close()

# Save live quadrupole strengths.
file = open('_output/live_fields_{}.dat'.format(scan_index), 'w')
for quad_id in controller.ind_quad_ids:
    field = controller.get_field(quad_id, 'live')
    file.write('{}, {}\n'.format(quad_id, field))
file.close()
    
# Save phases at each scan index.
file = open('output/phases.dat', 'w')
for (mux, muy) in phases:
    file.write('{}, {}\n'.format(mux, muy))
file.close()

# # Beam moments need to be measured using wire-scanner.
# # ...

exit()