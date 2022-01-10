"""Reconstruct covariance matrix at every node in the RTBT."""
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

from lib import analysis
from lib import optics
from lib.optics import TransferMatrixGenerator


# Using a single measurement (four wire-scanner profiles), reconstruct the covariance
# matrix at every node in the RTBT.
kinetic_energy = 0.8e9
filename = '_saved/2021-10-21/setting2/injturns400/turn400/WireAnalysisFmt-2021.10.21_19.22.35.pta.txt'
# filename = '_saved/2021-09-26/setting1/ramp_turns/profiles/WireAnalysisFmt-2021.09.27_00.10.34.pta.txt'
# filename = '_saved/2021-09-07/TBT_production_0.5ms/profiles/WireAnalysisFmt-2021.09.07_17.31.54.pta.txt'

measurement = analysis.Measurement(filename)
accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
tmatgen = TransferMatrixGenerator(sequence, kinetic_energy)

file = open('_output/data/rec_moments.dat', 'w')
file.write('node_id position sig_11 sig_12 sig_13 sig_14 sig_22 sig_23 sig_24 sig_33 sig_34 sig_44\n')
nodes = sequence.getNodes()
for rec_node in sequence.getNodes():
    print(rec_node.getId())
    moments_dict, tmats_dict = analysis.get_scan_info(measurement, tmatgen, rec_node.getId())
    moments_list, tmats_list = [], []
    for node_id in moments_dict:
        moments_list.extend(moments_dict[node_id])
        tmats_list.extend(tmats_dict[node_id])
    Sigma = analysis.reconstruct(tmats_list, moments_list, verbose=0)
    file.write('{} {:.2f} {} {} {} {} {} {} {} {} {} {}\n'.format(
        rec_node.getId(), 
        sequence.getDistanceBetween(nodes[0], rec_node),
        Sigma.get(0, 0), Sigma.get(0, 1), Sigma.get(0, 2), Sigma.get(0, 3), 
        Sigma.get(1, 1), Sigma.get(1, 2), Sigma.get(1, 3), 
        Sigma.get(2, 2), Sigma.get(2, 3), 
        Sigma.get(3, 3))
    )
file.close()


# Compute the model Twiss parameters. (The parameters at RTBT entrance are defined 
# by the closed orbit in the ring.)
pvl_data_source = PVLoggerDataSource(measurement.pvloggerid)

def get_seq_scenario(seq_name):
    sequence = accelerator.getComboSequence(seq_name)
    scenario = Scenario.newScenarioFor(sequence)
    scenario = pvl_data_source.setModelSource(sequence, scenario)
    scenario.resync()
    return sequence, scenario

# Get the model optics at the RTBT entrance.
sequence, scenario = get_seq_scenario('Ring')
tracker = AlgorithmFactory.createTransferMapTracker(sequence)
probe = ProbeFactory.getTransferMapProbe(sequence, tracker)
probe.setKineticEnergy(kinetic_energy)
scenario.setProbe(probe)
scenario.run()
trajectory = probe.getTrajectory()
calculator = CalculationsOnRings(trajectory)
state = trajectory.statesForElement('Begin_Of_Ring3')[0]
twiss_x, twiss_y, twiss_z = calculator.computeMatchedTwissAt(state)

# Track through the RTBT.
sequence, scenario = get_seq_scenario('RTBT')
tracker = AlgorithmFactory.createEnvelopeTracker(sequence)
tracker.setUseSpacecharge(False)
probe = ProbeFactory.getEnvelopeProbe(sequence, tracker)
probe.setBeamCurrent(0.0)
probe.setKineticEnergy(kinetic_energy)            
eps_x = eps_y = 20e-5 # [mm mrad] (arbitrary)
twiss_x = Twiss(twiss_x.getAlpha(), twiss_x.getBeta(), eps_x)
twiss_y = Twiss(twiss_y.getAlpha(), twiss_y.getBeta(), eps_y)
twiss_z = Twiss(0, 1, 0)
probe.initFromTwiss([twiss_x, twiss_y, twiss_z])
scenario.setProbe(probe)
scenario.run()
trajectory = probe.getTrajectory()
calculator = CalculationsOnBeams(trajectory)

file = open('_output/data/model_twiss.dat', 'w')
file.write('node_id position alpha_x alpha_y beta_x beta_y\n')
nodes = sequence.getNodes()
for node in nodes:
    state = trajectory.stateForElement(node.getId())
    twiss_x, twiss_y, _ = calculator.computeTwissParameters(state)
    print('Computing model Twiss parameters at {}'.format(node.getId()))
    file.write('{} {:.2f} {} {} {} {}\n'.format(
        node.getId(), 
        sequence.getDistanceBetween(nodes[0], node),
        twiss_x.getAlpha(), 
        twiss_y.getAlpha(), 
        twiss_x.getBeta(), 
        twiss_y.getBeta())
    )
file.close()

# Save the transfer matrices from each node to the target.
rec_node_id = 'RTBT:Tgt'
file = open('_output/data/transfer_mats.dat', 'w')
file.write('node_id position transfer_matrix_to_{}\n'.format(rec_node_id))
nodes = sequence.getNodes()
for node in nodes:
    M = tmatgen.generate(node.getId(), rec_node_id)
    file.write('{} {:.2f} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
        node.getId(), sequence.getDistanceBetween(nodes[0], node),
        *[M[i][j] for i in range(4) for j in range(4)])
    )        
file.close()
               
exit()