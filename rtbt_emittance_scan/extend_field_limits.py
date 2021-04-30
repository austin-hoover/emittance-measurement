"""
This script writes the default RTBT quadrupole power supply field limits and
field alarm limits to a file called 'default_field_limits.dat'. Each line
in the file gives the channel id and the default value separated by a comma, 
for example: 'RTBT_Mag:PS_QH02:B.LOLO, 2.145'. It then extends these limits.

Note: the script needs to be run with chief operator privileges.
"""
from xal.ca import Channel, ChannelFactory
from lib.phase_controller import PhaseController
from lib.helpers import loadRTBT

sequence = loadRTBT()
channel_factory = ChannelFactory.defaultFactory()
controller = PhaseController(sequence)

file = open('field_limits/default_field_limits.dat', 'w')
for ps_id in controller.ind_ps_ids:
    for key in ['B.LOLO', 'B.LOW', 'B.HIHI', 'B.HIGH', 
                'I.LOLO', 'I.LOW', 'I.HIHI', 'I.HIGH']:
        channel_id = ps_id + ':' + key
        print channel_id
        channel = channel_factory.getChannel(channel_id)
        channel.connectAndWait(0.1)
        print '  old = {:.3f}'.format(channel.getValFlt())
        file.write('{}, {}\n'.format(channel_id, channel.getValFlt()))
        if key in ['B.LOLO', 'B.LOW']:
            channel.putVal(0.0)
        elif key in ['B.HIHI', 'B.HIGH']:
            channel.putVal(30.0)
        elif key in ['I.LOLO', 'I.LOW']:
            channel.putVal(0.0)
        elif key in ['I.HIHI', 'I.HIGH']:
            channel.putVal(1000.0)
        print '  new = {:.3f}'.format(channel.getValFlt())
file.close()

exit()
