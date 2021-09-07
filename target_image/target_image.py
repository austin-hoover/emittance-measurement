from __future__ import print_function
import time
from xal.ca import Channel
from xal.ca import ChannelFactory

channel_factory = ChannelFactory.defaultFactory()
target_channel = channel_factory.getChannel('Target_Diag:TIS:Image')
target_channel.connectAndWait(0.1)

array = target_channel.getArrDbl() 


lt = time.localtime()
lt_string = '{}.{}.{}_{}.{}.{}'.format(lt.tm_year, lt.tm_mon, lt.tm_mday, 
                                       lt.tm_hour, lt.tm_min, lt.tm_sec)

filename = 'image_{}.dat'.format(lt_string)

file = open(filename, 'w')
for x in array:
    file.write(str(x) + ' ')
file.close()


exit()