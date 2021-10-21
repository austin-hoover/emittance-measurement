"""Save the longitudinal profile plot.."""
from __future__ import print_function
import sys
import os
import time
from datetime import datetime
from xal.ca import Channel
from xal.ca import ChannelFactory
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager

steps = 14

data = []

for i in range(steps):
    
    print('i = {}'.format(i))
    istr = str(i)
    if i < 10:
        istr = '0' + istr
    pv = 'Ring_Diag:BCM_D09:WF_Plot' + istr
    channel_factory = ChannelFactory.defaultFactory()
    channel = channel_factory.getChannel(pv)
    channel.connectAndWait()
        
    data.append(list(channel.getArrDbl()))
    
file = open('_output/data/bcm.dat', 'w')
for row in data:
    for x in row:
        file.write('{} '.format(x))
    file.write('\n')
file.close()
        
exit()