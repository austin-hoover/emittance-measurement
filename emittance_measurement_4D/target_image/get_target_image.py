from __future__ import print_function
import time
from datetime import datetime
from xal.ca import Channel
from xal.ca import ChannelFactory


class TargetImageGetter:
    
    def __init__(self):
        self.pv = 'Target_Diag:TIS:Image'
        self.channel_factory = ChannelFactory.defaultFactory()
        self.target_channel = self.channel_factory.getChannel(self.pv)
        self.target_channel.connectAndWait()
        
    def get_image(self):
        image = self.target_channel.getArrDbl() 
        now = datetime.utcnow()
        fstr = '{}.{}.{}_{}.{}.{}.{}'
        timestamp = fstr.format(now.year, now.month, now.day, 
                                now.hour, now.minute, now.second, 
                                now.microsecond)
        return image, timestamp
    
    def get_images(self, n=1, sleep_time=1.0):
        images, timestamps = [], []
        for i in range(n):
            print('Collecting image {}/{}'.format(i + 1, n))
            image, timestamp = self.get_image()
            images.append(image)
            timestamps.append(timestamp)
            if n > 1:
                time.sleep(sleep_time)
        return images, timestamps


    
ig = TargetImageGetter()

images, timestamps = ig.get_images(n=10, sleep_time=1.0)

file1 = open('_output/images.dat', 'w')
file2 = open('_output/timestamps.dat', 'w')

for image, timestamp in zip(images, timestamps):
    for x in image:
        file1.write(str(x) + ' ')
    file1.write('\n')
    file2.write(timestamp + '\n')

file1.close()
file2.close()
        
exit()