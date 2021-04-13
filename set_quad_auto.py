"""
This script changes one live quadrupole strength in the RTBT.
"""
import time
from xal.smf import Accelerator
from xal.smf.data import XMLDataManager
    
quad_id = 'RTBT_Mag:QH30'
frac_change = 0.02
    
accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
quad_node = sequence.getNodeWithId(quad_id)
B_init = quad_node.getField()
B_target = (1 + frac_change) * B_init
node.setField(B_target)
time.sleep(1.0)
B_final = node.getField()

print 'Initial field strength = {:.3f} [T/m]'.format(B_init)
print 'Desired field strength = {:.3f} [T/m]'.format(B_target)
print 'Final field strength = {:.3f} [T/m]'.format(B_final)