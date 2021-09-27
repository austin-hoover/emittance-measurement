"""Scan live phase advances at WS24, collect target image data."""
from __future__ import print_function
import time
import sys
import os
from pprint import pprint

from xal.ca import Channel
from xal.ca import ChannelFactory
from xal.extension.scan import WrappedChannel
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.sim.sync import SynchronizationException
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq
from xal.smf.data import XMLDataManager
from xal.smf.impl import MagnetMainSupply
from xal.tools.beam import CovarianceMatrix
from xal.tools.beam import PhaseVector
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib import utils
from lib.beam_trigger_lib import BeamTrigger

from get_target_image import TargetImageGetter


# Create channels
trigger_channel = WrappedChannel('ICS_Tim:Gate_BeamOn:SSMode')
trigger = BeamTrigger()
ig = TargetImageGetter()

    

# Read optics file.
file = open('_output/scan/optics.dat', 'r')
lines = [line.rstrip() for line in file]
quad_ids = lines[0].split()
fields_list = [[float(s) for s in line.split()] for line in lines[1:]]
    

# Perform the scan.
phase_controller = optics.PhaseController(kinetic_energy=0.8e9)

images = []
timestamps = []

    
for i, fields in enumerate(fields_list):
    print('i = {}'.format(i))
    
    # Set the machine optics.
    print('    Setting machine optics.')
    phase_controller.set_fields(quad_ids, fields, 'live')
    print('    Done.')
    
    # Pause?
    time.sleep(0.1)
    
    # Turn the beam on.
    trigger.makeShot()

    time.sleep(1.0)

    # Get the target image.
    image, timestamp = ig.get_image()
    images.append(image)
    timestamps.append(timestamp)
    
    # Turn the beam off?


# Save the data. 
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
