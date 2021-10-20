"""Get/save the target image."""
from __future__ import print_function
import sys
import os
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


def save_image_batch(images, filename):
    file = open(filename, 'w')
    for image in images:
        for x in image:
            file.write('{} '.format(x))
        file.write('\n')
    file.close()


if __name__ == '__main__':
    ig = TargetImageGetter()
    images, timestamps = ig.get_images(n=15, sleep_time=1.1)
    filename = '_output/data/image_{}.dat'.format(timestamps[0])
    save_image_batch(images, filename)
    exit()
