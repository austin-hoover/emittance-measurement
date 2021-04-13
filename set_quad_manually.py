"""
This script changes one live quadrupole strength in the RTBT by 
changing the book field limits.
"""
import time
from xal.ca import Channel, ChannelFactory
from xal.smf import Accelerator
from xal.smf.data import XMLDataManager
from lib.utils import arange
    
# Select quad to change
quad_id = 'RTBT_Mag:QH30'
quad_pv = 'RTBT_Mag:PS_QH30'
frac_change = 0.02
    
# Load accelerator and create channel factory
accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
factory = ChannelFactory.defaultFactory()

# Get and connect to B.LOLO and B.HIHI channels
lo = factory.getChannel(quad_pv + ':B.LOLO')
hi = factory.getChannel(quad_pv + ':B.HIHI')
lo.connectAndWait(1.0)
hi.connectAndWait(1.0)

# Store default B.LOLO and B.HIHI settings
BLOLO_init, BHIHI_init = lo.getValFlt(), hi.getValFlt()
print 'Initial B.LOLO, B.HIHI: {:.3f}, {:.3f}'.format(BLOLO_init, BHIHI_init)

# Change B.LOLO and B.HIHI settings
BLOLO_temp, BHIHI_temp = -100.0, +100.0
slow, step = False, 0.5
if slow:
    BLOLO_list = arange(BLOLO_init, BLOLO_temp, step)
    BHIHI_list = arange(BHIHI_init, BHIHI_temp, step)
    for BLOLO, BHIHI in zip(BLOLO_list, BHIHI_list):
        lo.putVal(BLOLO)
        hi.putVal(BHIHI)
else:
    lo.putVal(BLOLO_temp)
    hi.putVal(BHIHI_temp)
print 'Modified B.LOLO, B.HIHI: {:.3f}, {:.3f}'.format(lo.getValFlt(), hi.getValFlt())

# Change live quad strength
if False: 
    print 'Setting {} strength.'.format(quad_id)
    quad_node = sequence.getNodeWithId(quad_id)
    B_init = quad_node.getField()
    B_target = (1 + frac_change) * B_init
    node.setField(B_target)
    time.sleep(1.0)
    B_final = node.getField()
    print 'Initial field strength = {:.3f} [T/m]'.format(B_init)
    print 'Desired field strength = {:.3f} [T/m]'.format(B_target)
    print 'Final field strength = {:.3f} [T/m]'.format(B_final)
    
# Reset B.LOLO and B.HIHI settings to defaults
lo.putVal(BLOLO_init)
hi.putVal(BHIHI_init)
print 'Modified B.LOLO, B.HIHI: {:.3f}, {:.3f}'.format(lo.getValFlt(), hi.getValFlt())