from __future__ import print_function
import sys
import os

from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import analysis
from lib import optics


pvloggerid = 49547664
kinetic_energy = 0.8e9

phase_controller = optics.PhaseController(sync_mode='design', kinetic_energy=kinetic_energy)
phase_controller.sync_model_pvloggerid(pvloggerid)

# Read optics file.
file = open('_output/data/fields.dat', 'r')
lines = [line.rstrip() for line in file]
quad_ids = lines[0].split()
fields_list = [[float(s) for s in line.split()] for line in lines[1:]]
file.close()

# For each setting, extract the transfer matrix from each node to the target.
rec_node_id = 'RTBT:Tgt'
for i, fields in enumerate(fields_list):
    print('i = {}'.format(i))
    print('    Setting model optics.')
    phase_controller.set_fields(quad_ids, fields, 'model')
    
    file = open('_output/data/tmats_{}.dat'.format(i), 'w')
    file.write('node_id M11 M12 M13 M14 M21 M22 M23 M34 M31 M32 M33 M34 M41 M42 M43 M44\n')
    for node in phase_controller.sequence.getNodes():
#         print('    Getting M({} -> {})'.format(node.getId(), rec_node_id))
        M = phase_controller.transfer_matrix(node.getId(), rec_node_id)
        file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            node.getId(), *[M[i][j] for i in range(4) for j in range(4)])
        )
    file.close()
        
exit()