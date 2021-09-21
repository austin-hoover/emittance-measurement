"""Create MADX input files from OpenXAL model and PVLoggerID."""
from __future__ import print_function
from java.io import File
from xal.extension.extlatgen import AbstractDeviceDataSource
from xal.extension.extlatgen import MadXGenerator
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.smf.data import XMLDataManager
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings


pvloggerid = 49163506
kinetic_energy = 1e9
accelerator = XMLDataManager.loadDefaultAccelerator()
pvl_data_source = PVLoggerDataSource(pvloggerid)

def get_seq_scenario(seq_name):
    sequence = accelerator.getComboSequence(seq_name)
    scenario = Scenario.newScenarioFor(sequence)
    scenario = pvl_data_source.setModelSource(sequence, scenario)
    scenario.resync()
    return sequence, scenario

# Ring
#-------------------------------------------------------------------------------
# Get matched Twiss parameters at lattice entrance.
sequence, scenario = get_seq_scenario('Ring')
tracker = AlgorithmFactory.createTransferMapTracker(sequence)
probe = ProbeFactory.getTransferMapProbe(sequence, tracker)
probe.setKineticEnergy(kinetic_energy)
scenario.setProbe(probe)
scenario.run()
trajectory = probe.getTrajectory()
calculator = CalculationsOnRings(trajectory)
state = trajectory.stateForElement('Begin_Of_Ring1')
twiss_x, twiss_y, twiss_z = calculator.computeMatchedTwissAt(state)

print(calculator.computeFullTunes())
#
# # Initialize probe using matched optics.
# tracker = AlgorithmFactory.createEnvelopeTracker(sequence)
# tracker.setUseSpacecharge(False)
# probe = ProbeFactory.getEnvelopeProbe(sequence, tracker)
# probe.setBeamCurrent(0.0)
# probe.setKineticEnergy(kinetic_energy)
# eps_x = eps_y = 20e-5  # [mm mrad] (arbitrary)
# twiss_x = Twiss(twiss_x.getAlpha(), twiss_y.getBeta(), eps_x)
# twiss_y = Twiss(twiss_y.getAlpha(), twiss_y.getBeta(), eps_y)
# twiss_z = Twiss(0., 1., 0.)
# probe.initFromTwiss([twiss_x, twiss_y, twiss_z])
# scenario.setProbe(probe)
#
# # Generate MADX file.
# madx_generator = MadXGenerator([sequence], probe)
# data_source = AbstractDeviceDataSource.getPVLoggerDataSourceInstance(pvloggerid)
# # data_source = AbstractDeviceDataSource.getDesignDataSourceInstance()
# madx_generator.createMadInput(data_source, File('_output/ring/SNSring.madx'))

























#
# sequence = accelerator.getComboSequence('RTBT')
# tracker = AlgorithmFactory.createEnvelopeTracker(sequence)
# tracker.setUseSpacecharge(False)
# probe = ProbeFactory.getEnvelopeProbe(sequence, tracker)
# probe.setBeamCurrent(0.0)
# probe.setKineticEnergy(kinetic_energy)
# probe.initialize()
#
exit()