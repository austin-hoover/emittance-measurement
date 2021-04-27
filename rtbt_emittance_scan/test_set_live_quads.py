"""
This script changes the live quadrupole strengths in the RTBT and prints the
readback values to makes sure it worked.
"""
import time
import random
from lib.phase_controller import PhaseController, ind_quad_ids, all_quad_ids
from lib.phase_controller import init_twiss
from lib.helpers import loadRTBT


sequence = loadRTBT()
ref_ws_id = 'RTBT_Diag:WS24' # scan phases at this wire-scanner
init_twiss['ex'] = init_twiss['ey'] = 20e-6 # arbitrary [m*rad] 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Randomly change model quad strengths
for quad_id in ind_quad_ids:
    quad_strength = controller.get_field_strength(quad_id, 'model')
    delta_B = 0.05 * quad_strength
    controller.set_field_strength(quad_id, quad_strength + delta_B)
    
controller.sync_live_quads_with_model(all_quad_ids)

time.sleep(2.0)

print 'quadrupole id | model   | live'
print '---------------------------------'
for quad_id in all_quad_ids:
    B_model = controller.get_field_strength(quad_id, 'model')
    B_live = controller.get_field_strength(quad_id, 'live')
    print '{} | {:>7.4f} | {:.4f}'.format(quad_id, B_model, B_live)
