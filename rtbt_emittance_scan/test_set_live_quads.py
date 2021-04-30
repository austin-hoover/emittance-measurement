"""
This script changes the live quadrupole strengths in the RTBT and prints the
readback values to makes sure it worked.
"""
import time
from lib.phase_controller import PhaseController
from lib.helpers import loadRTBT
from lib.utils import multiply

frac_change = 0.1

sequence = loadRTBT()
controller = PhaseController(sequence)
ind_quad_ids = controller.ind_quad_ids

init_fields = controller.get_fields(ind_quad_ids, 'live')
target_fields = multiply(init_fields, (1 + frac_change))
    
controller.set_fields(ind_quad_ids, target_fields, 'live', 
                      max_change=1e6, wait=0.5, max_iters=10)

time.sleep(2.0)
final_fields = controller.get_fields(ind_quad_ids, 'live')

print 'quadrupole id | initial | target  |  final  | error'
print '------------------------------------------------------'
for i, quad_id in enumerate(ind_quad_ids):
    print '{} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.2e}'.format(
        quad_id, 
        init_fields[i], 
        target_fields[i], 
        final_fields[i],
        target_fields[i] - final_fields[i]
    )
    
exit()