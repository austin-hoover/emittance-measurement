"""Create MADX file from OpenXAL model."""

from java.io import File
from xal.extension.extlatgen import AbstractDeviceDataSource
from xal.extension.extlatgen import MadGenerator
from xal.extension.extlatgen import MadXGenerator
from xal.model.probe import Probe
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager

accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')

algorithm = AlgorithmFactory.createEnvelopeTracker(sequence)
algorithm.setUseSpacecharge(False) # might be unnecessary
probe = ProbeFactory.getEnvelopeProbe(sequence, algorithm)
probe.setBeamCurrent(0.0) # might be unnecessary

mad_generator = MadXGenerator([sequence], probe)
data_source = AbstractDeviceDataSource.getDesignDataSourceInstance()
mad_generator.createMadInput(data_source, File('rtbt.mad'))

exit()