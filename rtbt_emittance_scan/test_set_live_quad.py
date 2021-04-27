"""
This script changes one live quadrupole strength in the RTBT, then reads back
the value to verify that it worked.
"""
import time
from lib.phase_controller import PhaseController, init_twiss
from lib.helpers import loadRTBT


# Create phase controller
sequence = loadRTBT()
ref_ws_id = 'RTBT_Diag:WS24' # scan phases at this wire-scanner
init_twiss['ex'] = init_twiss['ey'] = 20e-6 # arbitrary [m*rad] 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Change single quad strength
quad_id = 'RTBT_Mag:QH02'
fractional_change = 0.1

init_field_strength_model = controller.get_field_strength(quad_id, 'model')
init_field_strength_live = controller.get_field_strength(quad_id, 'live')

target_field_strength = (1 + fractional_change) * init_field_strength_model
controller.set_field_strength(quad_id, target_field_strength)
controller.sync_live_quad_with_model(quad_id)

time.sleep(1.0)

final_field_strength_model = controller.get_field_strength(quad_id, 'model')
final_field_strength_live = controller.get_field_strength(quad_id, 'live')

print 'Init field strength  (model) = {} [T/m]'.format(init_field_strength_model)
print 'Init field strength  (live)  = {} [T/m]'.format(init_field_strength_live)
print 'Final field strength (model) = {} [T/m]'.format(final_field_strength_model)
print 'Final field strength (live)  = {} [T/m]'.format(final_field_strength_live)
