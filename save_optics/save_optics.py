"""
This script will save the ring and RTBT optics to a file. These will later be loaded into PyORBIT
"""
from xal.ca import Channel
from xal.ca import ChannelFactory
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager
from xal.smf.impl import MagnetMainSupply
from xal.tools.beam import Twiss
from xal.tools.beam import PhaseVector
from xal.tools.beam import CovarianceMatrix
from xal.tools.beam.calc import SimpleSimResultsAdaptor
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings


def save_mag_strengths(sequence, filename, live=False):
    scenario = Scenario.newScenarioFor(sequence)
    file = open(filename, 'w')
    for node in sequence.getNodesOfType('magnet'):
        if node.getType() in ['SH', 'SV']:
            continue
        if live:
            field = node.getField()
        else:
            field = scenario.elementsMappedTo(node)[0].getMagField()
        file.write('{}, {:.3f}\n'.format(node.getId(), field))
    file.close()
    

accelerator = XMLDataManager.loadDefaultAccelerator()
ring = accelerator.getComboSequence('Ring')
rtbt = accelerator.getComboSequence('RTBT')

save_mag_strengths(ring, 'ring_quads_model.dat', live=False)
save_mag_strengths(ring, 'ring_quads_live.dat', live=True)
save_mag_strengths(rtbt, 'rtbt_quads_model.dat', live=False)
save_mag_strengths(rtbt, 'rtbt_quads_live.dat', live=True)
             
exit()