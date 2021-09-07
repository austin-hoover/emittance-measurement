import time
from xal.ca import Channel
from xal.ca import ChannelFactory

channel_factory = ChannelFactory.defaultFactory()
target_channel = channel_factory.getChannel('Target_Diag:TIS:Image')
target_channel.connectAndWait(0.1)

array = target_channel.getArrDbl() 


current_time = time.strftime("%m.%d.%y %H:%M", time.localtime())
filename = 'image%s.txt' % current_time

file = open(filename, 'w')
for x in array:
    file.write(str(x) + ' ')
file.close()


exit()