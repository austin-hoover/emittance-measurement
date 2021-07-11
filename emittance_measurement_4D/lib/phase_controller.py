import math
import time
import warnings

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

# Local
from utils import subtract
from utils import norm
from utils import clip
from utils import put_angle_in_range
from helpers import get_trial_vals
from helpers import minimize
from helpers import list_from_xal_matrix
from lib.utils import radians
from lib.utils import linspace


# Available RTBT wire-scanners
ws_ids = ['RTBT_Diag:WS02', 'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
          'RTBT_Diag:WS23', 'RTBT_Diag:WS24']

# Twiss parameters at RTBT entrance
init_twiss = {
    'alpha_x': -1.378, 
    'alpha_y': 0.645,             
    'beta_x': 6.243, 
    'beta_y': 10.354, 
    'eps_x': 20e-6, # arbitrary
    'eps_y': 20e-6, # arbitrary
}

# Quadrupoles with independent power supplies
rtbt_ind_quad_ids = ['RTBT_Mag:QH02', 'RTBT_Mag:QV03', 'RTBT_Mag:QH04', 
                     'RTBT_Mag:QV05', 'RTBT_Mag:QH06', 'RTBT_Mag:QH12', 
                     'RTBT_Mag:QV13', 'RTBT_Mag:QH14', 'RTBT_Mag:QV15', 
                     'RTBT_Mag:QH16', 'RTBT_Mag:QV17', 'RTBT_Mag:QH18', 
                     'RTBT_Mag:QV19', 'RTBT_Mag:QH26', 'RTBT_Mag:QV27', 
                     'RTBT_Mag:QH28', 'RTBT_Mag:QV29', 'RTBT_Mag:QH30']


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
        self.kin_energy = kin_energy
        self.accelerator = XMLDataManager.loadDefaultAccelerator()
        self.sequence = self.accelerator.getComboSequence('RTBT')
        self.scenario = Scenario.newScenarioFor(self.sequence)
        self.scenario.setSynchronizationMode(Scenario.SYNC_MODE_LIVE)
        self.scenario.resync()
        self.algorithm = AlgorithmFactory.createEnvelopeTracker(self.sequence)
        self.algorithm.setUseSpacecharge(False)
        self.probe = ProbeFactory.getEnvelopeProbe(self.sequence, self.algorithm)
        self.probe.setBeamCurrent(0.0)
        self.probe.setKineticEnergy(self.kin_energy * 1e9)
        self.scenario.setProbe(self.probe)
        self.init_twiss = init_twiss
        self.track()
        
         # Get node for each RTBT quad and quad power supply.
        self.quad_nodes = [node for node in self.sequence.getNodesOfType('quad') 
                           if node.getId().startswith('RTBT') 
                           and not node.getId().endswith('QV01')]
        self.ps_nodes = [node.getMainSupply() for node in self.quad_nodes]
        self.quad_ids = node_ids(self.quad_nodes)
        self.ps_ids = node_ids(self.ps_nodes)
    
        # Get node and id of each independent RTBT quad and quad power supply.
        self.ind_quad_nodes, self.ind_ps_nodes = [], []
        for quad_node, ps_node in zip(self.quad_nodes, self.ps_nodes):
            if ps_node not in self.ind_ps_nodes:
                self.ind_ps_nodes.append(ps_node)
                self.ind_quad_nodes.append(quad_node)
        self.ind_quad_ids = node_ids(self.ind_quad_nodes)
        self.ind_ps_ids = node_ids(self.ind_ps_nodes)
            
        # Create dictionary of shared power supplies. Each key is an 
        # independent quad id, and each value is a list of quad ids which share
        # power with the indepent quad. We need this because the quads in the 
        # online model can be changed independently.
        self.shared_power = {}
        for quad_id, ps_id in zip(self.quad_ids, self.ps_ids):
            for ind_quad_id, ind_ps_id in zip(self.ind_quad_ids, self.ind_ps_ids):
                if ps_id == ind_ps_id and quad_id != ind_quad_id:
                    self.shared_power.setdefault(ind_quad_id, []).append(quad_id)

        # Connect to B_Book channels. These need to be changed at the same time as
        # the field settings, else the machine will trip.
        self.book_channels = {}
        for quad_id, ps_node in zip(self.quad_ids, self.ps_nodes):
            channel = ps_node.findChannel(MagnetMainSupply.FIELD_BOOK_HANDLE)
            self.book_channels[quad_id] = channel

        # Determine upper and lower bounds on power supplies.
        self.ps_lb, self.ps_ub = [], []
        for quad_node, ps_node in zip(self.ind_quad_nodes, self.ind_ps_nodes):
            lb = quad_node.toFieldFromCA(ps_node.lowerFieldLimit())
            ub = quad_node.toFieldFromCA(ps_node.upperFieldLimit())
            if lb > ub:
                lb, ub = ub, lb
            self.ps_lb.append(lb)
            self.ps_ub.append(ub)
            
        # Store the default field settings
        self.default_fields = self.get_fields(self.ind_quad_ids, 'model')    
        self.default_betas_at_target = self.beta_funcs('RTBT:Tgt')
        
    def set_kin_energy(self, kin_energy):
        """Set the probe kinetic energy [GeV]."""
        self.kin_energy = kin_energy
        self.probe.setKineticEnergy(1e9 * kin_energy)
        
    def initialize_envelope(self):
        """Reset the envelope probe to the start of the lattice."""
        self.scenario.resetProbe()
        self.probe.setKineticEnergy(1e9 * self.kin_energy)
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
    
    def twiss(self, node_id):
        """Return Twiss parameters at node entrance."""
        state = self.trajectory.statesForElement(node_id)[0]
        return compute_twiss(state, self.adaptor)
    
    def phases(self, node_id):
        """Return phase advances (mod 2pi) from start to node entrance."""
        return self.twiss(node_id)[:2]
    
    def beta_funcs(self, node_id):
        """Return beta functions at node entrance."""
        return self.twiss(node_id)[4:6]
        
    def transfer_matrix(self, node_id):
        """Return transfer matrix elements from start to node entrance."""
        scenario = Scenario.newScenarioFor(self.sequence)
        algorithm = AlgorithmFactory.createTransferMapTracker(self.sequence)
        probe = ProbeFactory.getTransferMapProbe(self.sequence, algorithm)
        scenario.setProbe(probe)
        scenario.run()
        state = probe.getTrajectory().statesForElement(node_id)[0]
        transfer_matrix = state.getTransferMap().getFirstOrder()
        transfer_matrix = list_from_xal_matrix(transfer_matrix)
        transfer_matrix_elements = [row[:4] for row in transfer_matrix[:4]]
        return transfer_matrix_elements
        
    def max_betas(self, start='RTBT_Mag:QH02', stop='RTBT_Diag:WS24'):
        """Return maximum x and y beta functions from start to stop node.
        
        Setting start=None starts tracks from the beginning of the lattice, and 
        setting stop=None tracks through the end of the lattice.
        """
        lo = None if start is None else self.trajectory.indicesForElement(start)[0]
        hi = None if stop is None else self.trajectory.indicesForElement(stop)[-1]
        beta_xs, beta_ys = [], []
        for params in self.tracked_twiss()[lo:hi]:
            mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y = params
            beta_xs.append(beta_x)
            beta_ys.append(beta_y)
        return max(beta_xs), max(beta_ys)

    def set_ref_ws_phases(self, mu_x, mu_y, beta_lims=(40, 40), verbose=0):
        """Set x and y phases from start to the reference wire-scanner. 
        
        TODO: It's currently is assumed that the reference wire-scanner is
        WS24. We therefore vary all but the last five quadrupoles to set the 
        phase advances, then vary the last five quadrupoles to set the beta
        functions at the target. If we want to choose a different wire-scanner, 
        then we'll need to only change the magnets that wire-scanner. The 
        downstream magnets should then be used to set the beta functions at the 
        target.

        Parameters
        ----------
        mu_x, mu_y : float
            The desired phase advances at the reference wire-scanner [rad].
        beta_lims : (xmax, ymax)
            Maximum beta functions to allow from QH02 to WS24.
        verbose : int
            If greater than zero, print a before/after summary.
            
        Returns
        -------
        fields : list[float]
            The correct field strengths for the independent quadrupoles.
        """        
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.beta_lims = beta_lims
                self.target_phases = [mu_x, mu_y]
                self.ref_ws_id = controller.ref_ws_id
                self.quad_ids = controller.ind_quad_ids[:-5]
                
            def score(self, trial, variables):
                fields = get_trial_vals(trial, variables)   
                self.controller.set_fields(self.quad_ids, fields, 'model')
                self.controller.track()
                calc_phases = self.controller.phases(self.ref_ws_id)
                residuals = subtract(calc_phases, self.target_phases)
                return norm(residuals) + self.penalty_function()
            
            def penalty_function(self):
                penalty = 0.
                for max_beta, beta_lim in zip(self.controller.max_betas(), self.beta_lims):
                    penalty += clip(max_beta - beta_lim, 0)
                return penalty**2
            
        scorer = MyScorer(self)
        var_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B12', 'B13', 
                     'B14', 'B15', 'B16', 'B17', 'B18', 'B19']
        bounds = (self.ps_lb[:-5], self.ps_ub[:-5])
        init_fields = self.default_fields[:-5]    
        self.restore_default_optics()
        fields = minimize(scorer, init_fields, var_names, bounds)
        if verbose > 0:
            print '  Desired phases : {:.3f}, {:.3f}'.format(mu_x, mu_y)
            print '  Calc phases    : {:.3f}, {:.3f}'.format(*self.phases(self.ref_ws_id))
            print '  Max betas (Q03 - WS24): {:.3f}, {:.3f}'.format(*self.max_betas())
            print '  Betas at target: {:.3f}, {:.3f}'.format(*self.beta_funcs('RTBT:Tgt'))
        return fields
    
    def constrain_size_on_target(self, max_beta_before_target=100., verbose=0):
        """Vary quads after WS24 to constrain beam size on target.
        
        Parameters
        ----------
        max_beta_before_target : float
            Maximum beta function to allow between ws24 and the target.
        verbose : int
            If greater than zero, print a before/after summary.
        """
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.quad_ids = controller.ind_quad_ids[-5:]
                
            def score(self, trial, variables):
                fields = get_trial_vals(trial, variables)  
                self.controller.set_fields(self.quad_ids, fields, 'model')
                self.controller.track()
                residuals = subtract(self.controller.default_betas_at_target, 
                                     self.controller.beta_funcs('RTBT:Tgt'))
                return norm(residuals) + self.penalty_function()
                
            def penalty_function(self):
                penalty = 0.
                for max_beta in self.controller.max_betas('RTBT_Diag:WS24', None):
                    penalty += clip(max_beta - max_beta_before_target, 0)          
                return penalty**2
            
        scorer = MyScorer(self)
        var_names = ['B26', 'B27', 'B28', 'B29', 'B30']
        init_fields = self.get_fields(self.ind_quad_ids[-5:], 'model')
        bounds = (self.ps_lb[-5:], self.ps_ub[-5:])
        fields = minimize(scorer, init_fields, var_names, bounds)
        self.set_fields(self.ind_quad_ids[-5:], fields, 'model')
        if verbose > 0:
            print '  Desired betas: {:.3f}, {:.3f}'.format(*self.default_betas_at_target)
            print '  Calc betas   : {:.3f}, {:.3f}'.format(*self.beta_funcs('RTBT:Tgt'))
            
    def get_phases_for_scan(self, phase_coverage=180.0, npts=3):
        """Create array of phases for scan. 
        
        Note: this resets model optics to default settings in order to 
        calculate the default phase advances around which to scan.

        Parameters
        ----------
        phase_coverages : float
            Range of phase advances to cover IN DEGREES. The phases are
            centered on the default phase. It is a pain because OpenXAL 
            computes the phases mod 2pi. 
        npts : int
            Total number of phases to include.
        """              
        def get_phases(phase, reverse=False):
            min_phase = put_angle_in_range(phase - 0.5 * radians(phase_coverage))
            max_phase = put_angle_in_range(phase + 0.5 * radians(phase_coverage))
            # Difference between and max phase is always <= 180 degrees
            abs_diff = abs(max_phase - min_phase)
            if abs_diff > math.pi:
                abs_diff = 2*math.pi - abs_diff
            # Return list of phases
            step = abs_diff / (npts - 1)
            phases = [min_phase]
            for _ in range(npts - 1):
                phase = put_angle_in_range(phases[-1] + step)
                phases.append(phase)
            if reverse:
                phases = phases[::-1]
            return phases
        
        self.restore_default_optics('model')
        mu_x0, mu_y0 = self.phases(self.ref_ws_id)
        x_phases = get_phases(mu_x0)
        y_phases = get_phases(mu_y0, reverse=True)
        return [(mu_x, mu_y) for mu_x, mu_y in zip(x_phases, y_phases)]

    def get_field(self, quad_id, opt='model'):
        """Return quadrupole field strength [T/m].
        
        quad_id : str
            Id of the quadrupole accelerator node.
        opt : {'model', 'live', 'book'}
            'model': value from online model
            'live' : live readback value 
            'book' : book setting
        """
        node = self.sequence.getNodeWithId(quad_id)
        if opt == 'model':
            return self.scenario.elementsMappedTo(node)[0].getMagField()
        elif opt == 'live':
            return node.getField()
        elif opt == 'book':
            return node.toFieldFromCA(self.book_channels[quad_id].getValFlt())
        else:
            raise ValueError("opt must be in {'model', 'live', 'book'}")
        
    def get_fields(self, quad_ids, opt='model'):
        """Return list of quadrupole field strengths [T/m]."""
        return [self.get_field(quad_id, opt) for quad_id in quad_ids]
    
    def set_field(self, quad_id, field, opt='model'):
        """Set quadrupole field strength [T/m].
        
        Note: this can lead to errors if the desired field is too far from the 
        book value.
        """
        node = self.sequence.getNodeWithId(quad_id)
        if opt == 'model':
            for elem in self.scenario.elementsMappedTo(node): 
                elem.setMagField(field)
            if quad_id in self.shared_power:
                for dep_quad_id in self.shared_power[quad_id]:
                    self.set_field(dep_quad_id, field)
        elif opt == 'live': 
            node.setField(field)
        elif opt == 'book':
            self.book_channels[quad_id].putVal(node.toCAFromField(field))
        else:
            raise ValueError("opt must be in {'model', 'live', 'book'}")
        
    def set_fields(self, quad_ids, fields, opt='model', max_frac_change=0.01, 
                   max_iters=100, sleep_time=0.5):
        """Set the fields of each quadrupole in the list.
        
        Note that the book values are always kept equal to the live values. 
        
        Parameters
        ----------
        quad_ids : list[str]
            List of quad ids to update.
        fields : list[float]
            List of new field strengths.
        opt : {'model', 'live', 'book'}
            Whether to change the model, live or book value.
        max_frac_change : float
            Maximum fractional field change. This is to ensure that no errors 
            are thrown when the book values are changed, which will occur if 
            the change is beyond 5%. We found 1% to be a safe value.
        max_iters : int
            Maximum iterations when stepping the quads. This is just so the while
            loop is guaranteed to terminate; it will never be approached.
        sleep_time : float
            Time to pause between field updates [seconds]. We found 0.5 seconds
            to be a safe value.
        """
        if opt == 'model':
            for quad_id, field in zip(quad_ids, fields):
                self.set_field(quad_id, field, 'model')
        elif opt == 'live':
            if sleep_time < 0.5:
                warnings.warn('sleep_time < 0.5 seconds... may trip MPS.')
            if max_frac_change > 0.01:
                warnings.warn('max_frac_change > 0.01... may trip MPS.')
            # Move all quad fields close enough to the desired values to 
            # avoid tripping MPS...
            stop, iters = False, 0
            while not stop and iters < max_iters:
                stop, iters = True, iters + 1                
                for quad_id, field in zip(quad_ids, fields):
                    book = self.get_field(quad_id, 'book')                    
                    change_needed = field - book
                    max_abs_change = max_frac_change * abs(book) 
                    if abs(change_needed) > max_abs_change:
                        stop = False
                        if change_needed >= 0.0:
                            new_field = book + max_abs_change
                        else:
                            new_field = book - max_abs_change                            
                        self.set_field(quad_id, new_field, 'book')
                        self.set_field(quad_id, new_field, 'live')
                time.sleep(sleep_time)
            #... and then set them to the desired values.
            for quad_id, field in zip(quad_ids, fields): 
                self.set_field(quad_id, field, 'book')
                self.set_field(quad_id, field, 'live')
                
    def restore_default_optics(self, opt='model', **kws):
        """Reset quadrupole fields to default values."""
        self.set_fields(self.ind_quad_ids, self.default_fields, opt, **kws)
        
    def sync_live_with_model(self, **kws):
        """Set the live quad fields to model values."""
        model_fields = self.get_fields(self.ind_quad_ids, 'model')
        self.set_fields(self.ind_quad_ids, model_fields, 'live', **kws)