"""This script prints the field limits for each RTBT quadrupole."""

from xal.ca import Channel, ChannelFactory
from xal.smf import Accelerator
from xal.smf.data import XMLDataManager

from lib.phase_controller import all_quad_ids, ind_quad_ids

accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
channel_factory = ChannelFactory.defaultFactory()

for quad_id in ind_quad_ids:
    quad_node = sequence.getNodeWithId(quad_id)
    quad_ps_id = quad_node.getMainSupply().getId()    
    print quad_id
    
    print 'Lower field limit = {:.3f}'.format(quad_node.lowerFieldLimit())
    print 'Lower alarm field limit = {:.3f}'.format(quad_node.lowerAlarmFieldLimit())
    print 'Lower display field limit = {:.3f}'.format(quad_node.lowerDisplayFieldLimit())
    print 'Lower warning field limit = {:.3f}'.format(quad_node.lowerWarningFieldLimit())
    for key in ['B.LOLO', 'B.LOW']:
        channel = channel_factory.getChannel(quad_ps_id + ':' + key)
        channel.connectAndWait(0.5)
        print '{} = {:.3f}'.format(key, channel.getValFlt())
        
    print 'Current live field strength = {:.3f} [T/m]'.format(quad_node.getField())
        
    print 'Upper field limit = {:.3f}'.format(quad_node.upperFieldLimit())
    print 'Upper alarm field limit = {:.3f}'.format(quad_node.upperAlarmFieldLimit())
    print 'Upper display field limit = {:.3f}'.format(quad_node.upperDisplayFieldLimit())
    print 'Upper warning field limit = {:.3f}'.format(quad_node.upperWarningFieldLimit())
    for key in ['B.HIHI', 'B.HIGH']:
        channel = channel_factory.getChannel(quad_ps_id + ':' + key)
        channel.connectAndWait(0.5)
        print '{} = {:.3f}'.format(key, channel.getValFlt())
    
    print ''
exit()
