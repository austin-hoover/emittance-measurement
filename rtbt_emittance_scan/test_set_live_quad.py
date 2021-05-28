"""
This script changes one live quadrupole strength in the RTBT, then reads back
the value to verify that it worked.
"""
import time
from lib.phase_controller import PhaseController, init_twiss
from lib.helpers import load_sequence


quad_id = 'RTBT_Mag:QH02'
frac_change = 0.1


sequence = load_sequence('RTBT')
controller = PhaseController(sequence)

init_field = controller.get_field(quad_id, 'live')
target_field = (1 + frac_change) * init_field
controller.set_field(quad_id, target_field, 'live')
time.sleep(1.0)
final_field = controller.get_field(quad_id, 'live')

print 'Initial = {} [T/m]'.format(init_field)
print 'Target  = {} [T/m]'.format(target_field)
print 'Final   = {} [T/m]'.format(final_field)

exit()