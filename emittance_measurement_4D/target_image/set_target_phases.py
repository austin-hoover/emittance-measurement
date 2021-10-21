"""Read optics from file, collect target image data.

Before running:
    * Make sure the optics file is in the correct location.
    * Make sure the output directory is correct.   
"""
from __future__ import print_function
import time
import sys
import os

from xal.ca import Channel
from xal.ca import ChannelFactory
from xal.extension.scan import WrappedChannel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib import utils
from lib.beam_trigger_lib import BeamTrigger

from get_target_image import TargetImageGetter


# Set up
trigger_channel = WrappedChannel('ICS_Tim:Gate_BeamOn:SSMode')
trigger = BeamTrigger()
ig = TargetImageGetter()
n_images_per_step = 5

# Read optics file.
file = open('_output/data/fields.dat', 'r')
lines = [line.rstrip() for line in file]
quad_ids = lines[0].split()
fields_list = [[float(s) for s in line.split()] for line in lines[1:]]
print('quad_ids =', quad_ids)

# Perform the scan.
phase_controller = optics.PhaseController()
images = []
timestamps = []

for i, fields in enumerate(fields_list):
    print('i = {}'.format(i))
    print('    Setting machine optics.')
    phase_controller.set_fields(quad_ids, fields, 'live')
    print('    Done.')

    # Pause?
    sleep_time = 0.1
    print('    Sleeping for {} seconds.'.format(sleep_time))
    time.sleep(sleep_time)

    for step in range(n_images_per_step):
        
        print('    Step {}/{}'.format(step, n_images_per_step))
        print('    Triggering beam.')
        trigger.makeShot()

        sleep_time = 1.1
        print('    Sleeping for {} seconds.'.format(sleep_time))
        time.sleep(1.1)

        print('    Collecting beam image on target.')
        image, timestamp = ig.get_image()
        images.append(image)
        timestamps.append(timestamp)

# Save the images.
file = open('_output/data/images.dat', 'w')
for image in images:
    for x in image:
        file.write(str(x) + ' ')
    file.write('\n')
file.close()

# Save the timestamps.
file = open('_output/data/timestamps.dat', 'w')
for timestamp in timestamps:
    file.write(timestamp + '\n')
file.close()
        
exit()
