"""
For each PTA output file, get the PV logger id, load the machine snapshot, 
change the beam energy if necessary, and compute the transfer matrices 
from the RTBT entrance to each wire-scanner.
"""
import collections
from pprint import pprint 

from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager
from xal.tools.beam.calc import CalculationsOnRings


ws_ids = ['RTBT_Diag:WS02', 'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
          'RTBT_Diag:WS23', 'RTBT_Diag:WS24']

filenames = [
    'data/ws/WireAnalysisFmt-2021.07.10_20.25.04.pta.txt',
    'data/ws/WireAnalysisFmt-2021.02.08_18.43.08.pta.txt',
    'data/ws/WireAnalysisFmt-2021.02.08_17.50.50.pta.txt',
    'data/ws/WireAnalysisFmt-2021.05.12_00.16.49.pta.txt',
    'data/ws/WireAnalysisFmt-2021.05.11_22.29.46.pta.txt',
    'data/ws/WireAnalysisFmt-2021.02.08_18.02.33.pta.txt',
    'data/ws/WireAnalysisFmt-2021.05.02_04.14.56.pta.txt',
    'data/ws/WireAnalysisFmt-2021.05.11_23.49.31.pta.txt',
    'data/ws/WireAnalysisFmt-2021.05.12_00.08.11.pta.txt',
    'data/ws/WireAnalysisFmt-2021.07.10_20.08.22.pta.txt',
    'data/ws/WireAnalysisFmt-2021.05.11_23.30.23.pta.txt',
    'data/ws/WireAnalysisFmt-2021.02.08_19.34.22.pta.txt',
    'data/ws/WireAnalysisFmt-2021.02.08_19.47.15.pta.txt',
    'data/ws/WireAnalysisFmt-2021.02.08_18.23.50.pta.txt',
    'data/ws/WireAnalysisFmt-2021.02.08_17.34.43.pta.txt',
    ]

def get_pvl_id(filename):
    file = open(filename, 'r')
    pvl_id = None
    for line in file:
        if line.startswith('PVLoggerID'):
            pvl_id = int(line.split('=')[1])
            return pvl_id
    file.close()
    
def get_kin_energy(filename):
    """Energy was 0.963 [GeV] before July, then switched to 1.0 [GeV]."""
    date, _ = filename.split('WireAnalysisFmt-')[-1].split('_')
    year, month, day = [int(token) for token in date.split('.')]
    kin_energy = 0.963e9 if month < 7 else 1e9
    return kin_energy
    
Info = collections.namedtuple('File', ['pvl_id', 'kin_energy'])
info_dict = dict()
for filename in filenames:
    pvl_id = get_pvl_id(filename)
    kin_energy = get_kin_energy(filename)
    info_dict[filename] = Info(pvl_id, kin_energy)
pprint(info_dict)


accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
scenario = Scenario.newScenarioFor(sequence)

def list_from_xal_matrix(matrix):
    """Return list of lists from XAL matrix object."""
    M = []
    for i in range(matrix.getRowCnt()):
        row = []
        for j in range(matrix.getColCnt()):
            row.append(matrix.getElem(i, j))
        M.append(row)
    return M

def transfer_matrix_elements(sequence, kin_energy, stop_node_id):
    """Return 16 transfer matrix elements from 'Begin_Of_RTBT1' to stop_node_id."""
    algorithm = AlgorithmFactory.createTransferMapTracker(sequence)
    probe = ProbeFactory.getTransferMapProbe(sequence, algorithm)
    probe.setKineticEnergy(kin_energy)
    scenario.setProbe(probe)
    scenario.run()
    trajectory = probe.getTrajectory()
    state = trajectory.stateForElement(stop_node_id)
    M = state.getTransferMap().getFirstOrder()
    M = list_from_xal_matrix(M)
    M = [row[:4] for row in M[:4]]
    elements = [elem for row in M for elem in row]
    return elements
    

for filename, info in info_dict.items():
    # None of the files from May have a PVLoggerID. I'll assume that the machine
    # state is the same as in February (the beam energy didn't change until July).
    if info.pvl_id < 0:
        info = info_dict[filenames[0]]
    # Load model from machine snapshot.
    pvl_data_source = PVLoggerDataSource(info.pvl_id)
    scenario = pvl_data_source.setModelSource(sequence, scenario)
    scenario.resync()
    # Save transfer matrix
    filename = filename.split('/')[-1]
    print filename
    file = open('data/transfer_matrix/model_transfer_mat_elems_default_Begin_Of_RTBT1_{}.dat'.format(filename), 'w')
    fstr = 16 * '{} ' + '\n'
    for ws_id in ws_ids:
        elements = transfer_matrix_elements(sequence, info.kin_energy, ws_id)
        file.write(fstr.format(*elements))
    file.close()
    
    ring = accelerator.getComboSequence('Ring')
    ring_scenario = Scenario.newScenarioFor(ring)
    ring_scenario = pvl_data_source.setModelSource(ring, ring_scenario)
    ring_scenario.resync()
    ring_tracker = AlgorithmFactory.createTransferMapTracker(ring)
    ring_probe = ProbeFactory.getTransferMapProbe(ring, ring_tracker)
    ring_probe.setKineticEnergy(info.kin_energy)
    ring_scenario.setProbe(ring_probe)
    ring_scenario.run()
    ring_trajectory = ring_probe.getTrajectory()
    ring_calculator = CalculationsOnRings(ring_trajectory)
    ring_state = ring_trajectory.statesForElement('Begin_Of_Ring3')[0]
    twiss_x, twiss_y, twiss_z = ring_calculator.computeMatchedTwissAt(ring_state)
    print 'alpha_x, alpha_y, beta_x, beta_y =', twiss_x.getAlpha(), twiss_y.getAlpha(), twiss_x.getBeta(), twiss_y.getBeta()

exit()