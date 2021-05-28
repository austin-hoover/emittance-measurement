"""
This script changes the live quadrupole strengths in the RTBT and prints the
readback values to makes sure it worked.
"""
import time
from lib.phase_controller import PhaseController
from lib.helpers import load_sequence
from lib.utils import multiply


frac_change = 0.1
field_set_kws = dict(max_frac_change=0.05, max_iters=100, sleep_time=0.1)


sequence = load_sequence('RTBT')
controller = PhaseController(sequence)
quad_ids = controller.ind_quad_ids

init_fields = controller.get_fields(quad_ids, 'live')
target_fields = multiply(init_fields, (1 + frac_change))
controller.set_fields(ind_quad_ids, target_fields, 'live', **field_set_kws)
time.sleep(1.0)
final_fields = controller.get_fields(quad_ids, 'live')


print 'quadrupole id | initial | target  |  final  | error'
print '------------------------------------------------------'
for i, quad_id in enumerate(quad_ids):
    print '{} | {:>7.4f} | {:>7.4f} | {:>7.4f} | {:>7.2e}'.format(
        quad_id, 
        init_fields[i], 
        target_fields[i], 
        final_fields[i],
        target_fields[i] - final_fields[i]
    ) 
exit()