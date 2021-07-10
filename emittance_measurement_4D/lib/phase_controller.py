from xal.ca import Channel
from xal.ca import ChannelFactory
from xal.extension.solver import Trial
from xal.extension.solver import Variable
from xal.extension.solver import Scorer
from xal.extension.solver import Stopper
from xal.extension.solver import Solver
from xal.extension.solver import Problem
from xal.extension.solver.ProblemFactory import getInverseSquareMinimizerProblem
from xal.extension.solver.SolveStopperFactory import maxEvaluationsStopper
from xal.extension.solver.algorithm import SimplexSearchAlgorithm
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


ws_ids = ['RTBT_Diag:WS02', 'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
          'RTBT_Diag:WS23', 'RTBT_Diag:WS24']

init_twiss = {
    'alpha_x': -1.378, 
    'alpha_y': 0.645,             
    'beta_x': 6.243, 
    'beta_y': 10.354, 
    'eps_x': 20e-6, # arbitrary
    'eps_y': 20e-6, # arbitrary
}


def node_ids(nodes):
    """Return list of node ids from list of accelerator nodes."""
    return [node.getId() for node in nodes]


def compute_twiss(state, adaptor):
    """Compute Twiss parameters from envelope trajectory state."""
    twiss_x, twiss_y, _ = adaptor.computeTwissParameters(state)
    alpha_x, beta_x = twiss_x.getAlpha(), twiss_x.getBeta()
    alpha_y, beta_y = twiss_y.getAlpha(), twiss_y.getBeta()
    eps_x, eps_y = twiss_x.getEmittance(), twiss_y.getEmittance()
    mu_x, mu_y, _ = adaptor.computeBetatronPhase(state).toArray()
    return mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y


class PhaseController:

    def __init__(self, ref_ws_id='RTBT_Diag:WS24', kin_energy=1.0):
        self.ref_ws_id = ref_ws_id
        self.accelerator = XMLDataManager.loadDefaultAccelerator()
        self.sequence = self.accelerator.getComboSequence('RTBT')
        self.scenario = Scenario.newScenarioFor(self.sequence)
        self.scenario.setSynchronizationMode(Scenario.SYNC_MODE_LIVE)
        self.scenario.resync()
        self.algorithm = AlgorithmFactory.createEnvelopeTracker(self.sequence)
        self.algorithm.setUseSpacecharge(False)
        self.probe = ProbeFactory.getEnvelopeProbe(self.sequence, self.algorithm)
        self.probe.setBeamCurrent(0.0)
        self.probe.setKineticEnergy(kin_energy * 1e9)
        self.scenario.setProbe(self.probe)
        self.init_twiss = init_twiss
        self.track()

    def initialize_envelope(self):
        """Reset the envelope probe to the start of the lattice."""
        self.scenario.resetProbe()
        twissX = Twiss(init_twiss['alpha_x'], init_twiss['beta_x'], init_twiss['eps_x'])
        twissY = Twiss(init_twiss['alpha_y'], init_twiss['beta_y'], init_twiss['eps_y'])
        twissZ = Twiss(0, 1, 0)
        Sigma = CovarianceMatrix().buildCovariance(twissX, twissY, twissZ)
        self.probe.setCovariance(Sigma)
        
    def track(self):
        """Return envelope trajectory through the lattice."""
        self.initialize_envelope()
        self.scenario.run()
        self.trajectory = self.probe.getTrajectory()
        self.adaptor = SimpleSimResultsAdaptor(self.trajectory) 
        self.states = self.trajectory.getStatesViaIndexer()
        self.positions = [state.getPosition() for state in self.states]
        return self.trajectory
    
    def tracked_twiss(self):
        """Return Twiss parameters at each state in trajectory."""
        return [compute_twiss(state, self.adaptor) for state in self.states]
        
        
        

        
        
        
        
        
        
# class TestButtonListener(ActionListener):
#     def __init__(self, phase_controller):
#         self.phase_controller = phase_controller

#     def actionPerformed(self, actionEvent):
#         energy = float(self.phase_controller.energy_text_field.getText())
#         print 'Energy = {}'.format(energy)