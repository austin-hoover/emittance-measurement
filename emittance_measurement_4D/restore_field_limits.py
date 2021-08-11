"""
This script restores the default RTBT quadrupole power supply field limits and
field alarm limits to their default values. The default values are read from a
file called 'default_field_limits.dat'. Each line in the file gives the channel
id and the default value separated by a comma, for example: 
'RTBT_Mag:PS_QH02:B.LOLO, 2.145'. 

Note: the script needs to be run with operator privileges.
"""
import time
from xal.ca import Channel, ChannelFactory
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager


accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
channel_factory = ChannelFactory.defaultFactory()

file = open('field_limits/default_field_limits.dat', 'r')
for line in file:
    channel_id, value = line.rstrip().split(',')
    channel = channel_factory.getChannel(channel_id)
    channel.connectAndWait(0.1)
    print channel_id
    print '  old = {:.3f}'.format(channel.getValFlt())
    channel.putVal(value)
    print '  new = {:.3f}'.format(channel.getValFlt())
file.close()
exit()