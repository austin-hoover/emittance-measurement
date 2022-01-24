"""This script scans the phase advances at the target and saves the beam image
at each optics setting.

The optics are not computed here; instead, they are read from a file. To avoid
memory errors, each image is saved to a separate file.

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

from get_target_image import TargetImageGetter
from get_target_image import save_image_batch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib import utils
from lib.beam_trigger_lib import BeamTrigger


# Setup
trigger = BeamTrigger()
ig = TargetImageGetter()
n_images_per_step = 5 # Average over this many images at each optics setting.

# Read optics file.
file = open('_output/data/fields.dat', 'r')
lines = [line.rstrip() for line in file]
quad_ids = lines[0].split()
fields_list = [[float(s) for s in line.split()] for line in lines[1:]]
print('quad_ids =', quad_ids)

# Create the PhaseController (Note: the kinetic energy does not need to be 
# specified; all the PhaseController is doing is setting quadrupole strengths.)
phase_controller = optics.PhaseController(kinetic_energy=0.8e9)

# Perform the scan.
for i, fields in enumerate(fields_list):
    print('i = {}'.format(i))
    
    # Set the machine optics.
    print('  Setting machine optics.')
    phase_controller.set_fields(quad_ids, fields, 'live')
    print('  Done.')

    # Pause.
    sleep_time = 0.1
    print('  Sleeping for {} seconds.'.format(sleep_time))
    time.sleep(sleep_time)

    # Collect an image batch.
    images, timestamps = [], []
    for step in range(n_images_per_step):
        # Send a beam pulse to the target.
        print('  Step {}/{}'.format(step, n_images_per_step))
        print('  Triggering beam.')
        trigger.fire()
        # Pause.
        sleep_time = 1.5
        print('  Sleeping for {} seconds.'.format(sleep_time))
        time.sleep(sleep_time)
        # Collect beam image on the target.
        print('  Collecting beam image on target.')
        image, timestamp = ig.get_image()
        images.append(image)
        timestamps.append(timestamp)
    
    filename = '_output/data/image_{}.dat'.format(timestamps[0])
    save_image_batch(images, filename)
        
exit()
