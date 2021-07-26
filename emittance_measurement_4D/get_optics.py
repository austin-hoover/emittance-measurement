from pprint import pprint

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

from lib.helpers import list_from_xal_matrix


accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
scenario = Scenario.newScenarioFor(sequence)
algorithm = AlgorithmFactory.createTransferMapTracker(sequence)
probe = ProbeFactory.getTransferMapProbe(sequence, algorithm)
scenario.setProbe(probe)
scenario.run()


# Set start and stop nodes.
start_node_id = sequence.getNodes()[0].getId()
stop_node_id = 'RTBT_Diag:WS24'

# Check if the nodes are in order.
reverse = False
node_ids = [node.getId() for node in sequence.getNodes()]
if node_ids.index(start_node_id) > node_ids.index(stop_node_id):
    start_node_id, stop_node_id = stop_node_id, start_node_id
    reverse = True
    
print 'reverse =', reverse

# Get transfer matrix from upstream to downstream node.
trajectory = probe.getTrajectory()
state1 = trajectory.stateForElement(start_node_id)
state2 = trajectory.stateForElement(stop_node_id)
M1 = state1.getTransferMap().getFirstOrder()
M2 = state2.getTransferMap().getFirstOrder()
M = M2.times(M1.inverse())

if reverse:
    M = M.inverse()
        
for row in list_from_xal_matrix(M)[:4]:
    print row[:4]
    
exit()