"""Scan live phase advances at WS24, collect target image data."""
from __future__ import print_function
import sys
import os
from pprint import pprint

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


# Read optics file.
file = open('_output/optics.dat', 'r')
lines = [line.rstrip() for line in file]
quad_ids = lines[0].split()
fields_list = [[float(s) for s in line.split()] for line in lines[1:]]
    

# Perform the scan.
controller = optics.PhaseController()
    
for i, fields in enumerate(fields_list):
    
    print('i = {}'.format(i))
    
    # Set the machine optics.
    controller.set_fields(quad_ids, fields, 'live')
    
    # Pause?
    
    
    # Turn the beam on.
    
    
    # Collect target image data.
    
    
    # Turn the beam off.
    

exit()