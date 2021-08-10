"""Create MADX file from OpenXAL model."""

from java.io import File
from xal.extension.extlatgen import AbstractDeviceDataSource
from xal.extension.extlatgen import MadXGenerator
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
from xal.tools.beam import CovarianceMatrix
from xal.tools.beam import PhaseVector
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings


pvloggerid = 48842340
kin_energy = 1e9

accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
tracker = AlgorithmFactory.createEnvelopeTracker(sequence)
tracker.setUseSpacecharge(False) 
probe = ProbeFactory.getEnvelopeProbe(sequence, tracker)
probe.setBeamCurrent(0.0) 
probe.setKineticEnergy(kin_energy)
probe.initialize()

sequence = accelerator.getComboSequence('RTBT')
mad_generator = MadXGenerator([sequence], probe)
# data_source = AbstractDeviceDataSource.getPVLoggerDataSourceInstance(pvloggerid)
data_source = AbstractDeviceDataSource.getDesignDataSourceInstance()
mad_generator.createMadInput(data_source, File('rtbt.madx'))

exit()