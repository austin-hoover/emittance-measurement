from __future__ import print_function
import sys
import os
import time
from datetime import datetime
from xal.ca import Channel
from xal.ca import ChannelFactory


class TargetImageGetter:
    """Class to retrieve images of the beam on the target."""
    def __init__(self):
        self.pv = 'Target_Diag:TIS:Image'
        self.channel_factory = ChannelFactory.defaultFactory()
        self.target_channel = self.channel_factory.getChannel(self.pv)
        self.target_channel.connectAndWait()

    def get_image(self):
        """Return the current image and timestamp.
        
        Returns
        -------
        image : list, shape (80000,)
            The target image array.
        timestamp : str
            Current time in format: year.month.day_hour.minute.second.microsecond.
            Example: '2021.10.21_23.28.36.569000'.
        """
        image = self.target_channel.getArrDbl()
        now = datetime.utcnow()
        fstr = '{}.{}.{}_{}.{}.{}.{}'
        timestamp = fstr.format(now.year, now.month, now.day, 
                                now.hour, now.minute, now.second, 
                                now.microsecond)
        return image, timestamp
    
    def get_images(self, n=1, sleep_time=1.0):
        """Return list of `n` images and timestamps with `sleep_time` 
        seconds pause between images."""
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
    """Save multiple image arrays to a single file.
    
    Parameters
    ----------
    images : list[list, shape (80000,)]
        A list of image arrays.
    filename : str
        The name of the saved file.
    """
    file = open(filename, 'w')
    for image in images:
        for x in image:
            file.write('{} '.format(x))
        file.write('\n')
    file.close()


if __name__ == '__main__':
    # Get and save a batch of images to a time-stamped file.
    ig = TargetImageGetter()
    images, timestamps = ig.get_images(n=15, sleep_time=1.75)
    filename = '_output/data/image_{}.dat'.format(timestamps[0])
    save_image_batch(images, filename)
    exit()