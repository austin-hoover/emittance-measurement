"""Compute and print the model ring Twiss parameters at the injection point."""
from __future__ import print_function
import math

from xal.model.probe import Probe
from xal.model.probe import TransferMapProbe
from xal.model.probe.traj import Trajectory
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq
from xal.smf.data import XMLDataManager
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnRings


kin_energy = 0.8e9 # [eV]
ypmax = 1.7 # [mrad]
sync_live = False
#pvloggerid = None
pvloggerid = 49548117

accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('Ring')
scenario = Scenario.newScenarioFor(sequence)

if sync_live:
    scenario.setSynchronizationMode(Scenario.SYNC_MODE_LIVE)
    scenario.resync()
    
if pvloggerid is not None:
    pvl_data_source = PVLoggerDataSource(pvloggerid)
    scenario = pvl_data_source.setModelSource(sequence, scenario)
    scenario.resync()

    
print('Kinetic energy = {:.3e} [eV]'.format(kin_energy))
print('ypmax = {} [mrad]'.format(ypmax))
print('Sync live = {}'.format(sync_live))
print('pvloggerid = {}'.format(pvloggerid))
    

algorithm = AlgorithmFactory.createTransferMapTracker(sequence)
probe = ProbeFactory.getTransferMapProbe(sequence, algorithm)
probe.setKineticEnergy(kin_energy)
scenario.setProbe(probe)
scenario.run()
trajectory = scenario.getTrajectory()

calculator = CalculationsOnRings(trajectory)
state = trajectory.stateForElement('Ring_Inj:Foil')
twissX, twissY, _ = calculator.computeMatchedTwissAt(state)
tunes = calculator.computeFullTunes()
nux, nuy = tunes.getx(), tunes.gety()
alpha_x = twissX.getAlpha()
alpha_y = twissY.getAlpha()
beta_x = twissX.getBeta()
beta_y = twissY.getBeta()
print('Ring tunes = {}, {}'.format(nux, nuy))
print('Twiss parameters at foil:')
print('  alpha_x = {} [rad]'.format(alpha_x))
print('  alpha_y = {} [rad]'.format(alpha_y))
print('  beta_x = {} [m/rad]'.format(beta_x))
print('  beta_y = {} [m/rad]'.format(beta_y))


gamma_x = (1 + alpha_x**2) / beta_x
ratio = math.sqrt(beta_y / gamma_x)
print('To paint equal emittances with x-yp painting:')
print('  xmax(xp=0) / ypmax(y=0) = {}'.format(ratio))
print('  If ypmax(y=0) = {} [mrad], then xmax(xp=0) = {} [mm]'.format(ypmax, ratio * ypmax))

exit()
