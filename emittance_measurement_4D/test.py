"""
This script will get the ring Twiss parameters at the entrance of the RTBT.
"""
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager
from xal.tools.beam import Twiss
from xal.tools.beam import PhaseVector
from xal.tools.beam import CovarianceMatrix
from xal.tools.beam.calc import SimpleSimResultsAdaptor
from xal.tools.beam.calc import CalculationsOnBeams, CalculationsOnRings


kin_energy

# Load SNS ring
accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('Ring')
scenario = Scenario.newScenarioFor(sequence)
algorithm = AlgorithmFactory.createTransferMapTracker(sequence)
probe = ProbeFactory.getTransferMapProbe(sequence, algorithm)
probe.setKineticEnergy(kin_energy)
scenario.setProbe(probe)
scenario.run()
trajectory = probe.getTrajectory()

# Get ring Twiss parameters at injection point
calculator = CalculationsOnRings(trajectory)
twiss_x, twiss_y, twiss_z = calculator.ringMatchedTwiss()
alpha_x, beta_x = twiss_x.getAlpha(), twiss_x.getBeta()
alpha_y, beta_y = twiss_y.getAlpha(), twiss_y.getBeta()
print twiss_x
print twiss_y

# Get Twiss parameters at RTBT entrance
scenario = Scenario.newScenarioFor(sequence)
algorithm = AlgorithmFactory.createEnvelopeTracker(sequence)
algorithm.setUseSpacecharge(False)
probe = ProbeFactory.getEnvelopeProbe(sequence, algorithm)
probe.setBeamCurrent(0.0)
probe.setKineticEnergy(kin_energy)

twiss_x.setTwiss(alpha_x, beta_x, 20e-6)
twiss_y.setTwiss(alpha_y, beta_y, 20e-6)
twiss_z.setTwiss(0, 1, 0)


Sigma = CovarianceMatrix().buildCovariance(twiss_x, twiss_y, twiss_z)
probe.setCovariance(Sigma)
scenario.setProbe(probe)
scenario.run()
trajectory = probe.getTrajectory()
adaptor = SimpleSimResultsAdaptor(trajectory) 

stop_node_id = 'Begin_Of_Ring3' # same as 'Begin_Of_RTBT'
state = trajectory.statesForElement(stop_node_id)[0]
twiss_x, twiss_y, _ = adaptor.computeTwissParameters(state)

print 'alpha_x, alpha_y =', twiss_x.getAlpha(), twiss_y.getAlpha()
print 'beta_x, beta_y =', twiss_x.getBeta(), twiss_y.getBeta()