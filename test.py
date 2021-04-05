"""
This script changes one live quadrupole strength in the RTBT, then reads back
the value to verify that it worked.
"""
from lib.phase_controller import PhaseController
from lib.utils import loadRTBT, init_twiss
import time
from xal.ca import Channel, ChannelFactory

# Create phase controller
sequence = loadRTBT()
ref_ws_id = 'RTBT_Diag:WS24' # scan phases at this wire-scanner
init_twiss['ex'] = init_twiss['ey'] = 20e-6 # arbitrary [m*rad] 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Change single quad strength
quad_id = 'RTBT_Mag:QV15'
quad_pv = 'RTBT_Mag:PS_QV15'

node = sequence.getNodeWithId(quad_id)
factory = ChannelFactory.defaultFactory()

channel_hi = factory.getChannel(quad_pv + ':B.HIHI')
channel_hi.connectAndWait(1.0)
channel_lo = factory.getChannel(quad_pv + ':B.LOLO')
channel_lo.connectAndWait(1.0)
init_lo, init_hi = channel_lo.getValFlt(), channel_hi.getValFlt()

channel_hi.putVal(100.0)
channel_lo.putVal(-100.0)

time.sleep(3.0)

print 'modified:', channel_lo.getValFlt(), channel_hi.getValFlt()

time.sleep(3.0)

channel_lo.putVal(init_lo)
channel_hi.putVal(init_hi)

time.sleep(3.0)

print 'reset:', channel_lo.getValFlt(), channel_hi.getValFlt()
