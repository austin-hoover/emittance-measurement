"""
This module is for controlling the phase advances at a wire-scanner in the
RTBT. 
"""

import math
import time
from xal.ca import Channel, ChannelFactory
from xal.smf import Accelerator, AcceleratorSeq 
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory, ProbeFactory, Scenario
from xal.tools.beam import Twiss, PhaseVector, CovarianceMatrix
from xal.tools.beam.calc import SimpleSimResultsAdaptor, CalculationsOnBeams
from xal.extension.solver import Trial, Variable, Scorer, Stopper, Solver, Problem
from xal.extension.solver.ProblemFactory import getInverseSquareMinimizerProblem
from xal.extension.solver.SolveStopperFactory import maxEvaluationsStopper
from xal.extension.solver.algorithm import SimplexSearchAlgorithm

from utils import subtract, norm, step_func, put_angle_in_range
from helpers import get_trial_vals, minimize
from lib.utils import linspace


ws_ids = ['RTBT_Diag:WS02', 'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
          'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
init_twiss = {'ax': -1.378, 'ay':0.645, 'bx': 6.243, 'by':10.354, 
              'ex': 20e-6, 'ey': 20e-6} 
design_betas_at_target = (57.705, 7.909) 


def get_ids(nodes):
    return [node.getId() for node in nodes]


class PhaseController:
    """Class to control phases at a wire-scanner in the RTBT."""
    def __init__(self, sequence, ref_ws_id=ws_ids[0], init_twiss=init_twiss):
        """Constructor.
        
        Parameters
        ----------
        sequence : AcceleratorSeq
            Sequence representing the RTBT.
        ref_ws_id : str
            Node id of reference wire-scanner.
        init_twiss : dict
            Dictionary containing 'ax', 'ay', 'bx', 'by', 'ex', 'ey'.
        """
        self.ref_ws_id = ref_ws_id
        self.sequence = sequence
        self.scenario = Scenario.newScenarioFor(sequence)
        self.algorithm = AlgorithmFactory.createEnvelopeTracker(sequence)
        self.algorithm.setUseSpacecharge(False)
        self.probe = ProbeFactory.getEnvelopeProbe(sequence, self.algorithm)
        self.probe.setBeamCurrent(0.0)
        self.scenario.setProbe(self.probe)
        self.init_twiss = init_twiss
        self.track()
        
        # Get node for each RTBT quad and quad power supply
        self.quad_nodes = [node for node in sequence.getNodesOfType('quad') 
                           if node.getId().startswith('RTBT') and not node.getId().endswith('QV01')]
        self.ps_nodes = [node.getMainSupply() for node in self.quad_nodes]
        self.quad_ids = get_ids(self.quad_nodes)
        self.ps_ids = get_ids(self.ps_nodes)
    
        # Get node and id of each independent RTBT quad and quad power supply
        self.ind_quad_nodes, self.ind_ps_nodes = [], []
        for quad_node, ps_node in zip(self.quad_nodes, self.ps_nodes):
            if ps_node not in self.ind_ps_nodes:
                self.ind_ps_nodes.append(ps_node)
                self.ind_quad_nodes.append(quad_node)
        self.ind_quad_ids = get_ids(self.ind_quad_nodes)
        self.ind_ps_ids = get_ids(self.ind_ps_nodes)
        
        # Create dictionary of shared power supplies. Each key is an 
        # independent quad id, and each value is a list of quad ids who share
        # power with the indepent quad. We need this because the quads in the 
        # online model can be changed independently.
        self.shared_ps_dict = {}
        for quad_id, ps_id in zip(self.quad_ids, self.ps_ids):
            for ind_quad_id, ind_ps_id in zip(self.ind_quad_ids, self.ind_ps_ids):
                if ps_id == ind_ps_id and quad_id != ind_quad_id:
                    self.shared_ps_dict.setdefault(ind_quad_id, []).append(quad_id)
                    
        # Connect to B_Book channels
        self.book_channels = {}
        cfactory = ChannelFactory.defaultFactory()
        for quad_id, ps_id in zip(self.quad_ids, self.ps_ids):
            channel = cfactory.getChannel(ps_id + ':B_Book')
            channel.connectAndWait(0.1)
            self.book_channels[quad_id] = channel
        
        # Bounds on power supplies
        self.ps_lb, self.ps_ub = [], []
        for quad_node, ps_node in zip(self.ind_quad_nodes, self.ind_ps_nodes):
            lb = quad_node.toFieldFromCA(ps_node.lowerFieldLimit())
            ub = quad_node.toFieldFromCA(ps_node.upperFieldLimit())
            if lb > ub:
                lb, ub = ub, lb
            self.ps_lb.append(lb)
            self.ps_ub.append(ub)
            
        self.default_fields = self.get_fields(self.ind_quad_ids, 'model')
        
    def get_node(self, node_id):
        return self.sequence.getNodeWithId(node_id)
    
    def restore_default_optics(self, live=False):
        self.set_fields(self.ind_quad_ids, self.default_fields, 'model')
        if live:
            self.set_fields(self.ind_quad_ids, self.default_fields, 'live')

    def initialize_envelope(self):
        self.scenario.resetProbe()
        ax, bx, ex = [self.init_twiss[key] for key in ('ax', 'bx', 'ex')]
        ay, by, ey = [self.init_twiss[key] for key in ('ay', 'by', 'ey')]
        twissX = Twiss(ax, bx, ex)
        twissY = Twiss(ay, by, ey)
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
        
    def get_twiss(self):
        """Get Twiss parameters at every state in the trajectory."""
        twiss = []
        for state in self.states:
            twissX, twissY, _ = self.adaptor.computeTwissParameters(state)
            ax, bx = twissX.getAlpha(), twissX.getBeta()
            ay, by = twissY.getAlpha(), twissY.getBeta()
            ex, ey = twissX.getEmittance(), twissY.getEmittance()
            nux, nuy, _ = self.adaptor.computeBetatronPhase(state).toArray()
            twiss.append([nux, nuy, ax, ay, bx, by, ex, ey])
        return twiss
    
    def get_moments_at(self, node_id):
        state = self.trajectory.statesForElement(node_id)[0]
        Sigma = state.getCovarianceMatrix()
        return [Sigma.getElem(0, 0), Sigma.getElem(2, 2), Sigma.getElem(0, 2)]
    
    def get_transfer_matrix_at(self, node_id):
        scenario = Scenario.newScenarioFor(self.sequence)
        algorithm = AlgorithmFactory.createTransferMapTracker(self.sequence)
        probe = ProbeFactory.getTransferMapProbe(self.sequence, algorithm)
        scenario.setProbe(probe)
        scenario.run()
        trajectory = probe.getTrajectory()
        states = trajectory.getStatesViaIndexer()
        state = trajectory.statesForElement(node_id)[0]
        transfer_matrix = state.getTransferMap().getFirstOrder()
        M = []
        for i in range(4):
            row = []
            for j in range(4):
                row.append(transfer_matrix.getElem(i, j))
            M.append(row)
        return M
        
    def get_max_betas(self, start='RTBT_Mag:QH02', stop='RTBT_Diag:WS24'):
        lo = self.trajectory.indicesForElement(start)[0]
        hi = -1 if stop is None else self.trajectory.indicesForElement(stop)[-1]
        twiss = self.get_twiss()
        bx_list, by_list = [], []
        for (nux, nuy, ax, ay, bx, by, ex, ey) in twiss[lo:hi]:
            bx_list.append(bx)
            by_list.append(by)
        return max(bx_list), max(by_list)
    
    def get_betas_at_target(self):
        return self.get_twiss()[-1][4:6]
    
    def get_field(self, quad_id, opt='model'):
        """Return quadrupole field strength [T/m].
        
        quad_id : str
            Id of the quadrupole accelerator node.
        opt : {'model', 'live', 'book'}
            'model': model value
            'live' : live readback value from EPICS
            'book' : book setting
            
        The same parameters are used in the next few functions.
        """
        node = self.get_node(quad_id)
        if opt == 'model':
            return self.scenario.elementsMappedTo(node)[0].getMagField()
        elif opt == 'live':
            return node.getField()
        elif opt == 'book':
            return node.toFieldFromCA(self.book_channels[quad_id].getValFlt())
            
    def get_fields(self, quad_ids, opt='model'):
        return [self.get_field(quad_id, opt) for quad_id in quad_ids]
    
    def set_field(self, quad_id, field, opt='model'):
        node = self.sequence.getNodeWithId(quad_id)
        if opt == 'model':
            for elem in self.scenario.elementsMappedTo(node): 
                elem.setMagField(field)
            if quad_id in self.shared_ps_dict:
                for dep_quad_id in self.shared_ps_dict[quad_id]:
                    self.set_field(dep_quad_id, field)
        elif opt == 'live': 
            # This also changes the book value
            self.book_channels[quad_id].putVal(node.toCAFromField(field))
            node.setField(field)

    def set_fields(self, quad_ids, fields, opt='model', max_change=1e6, 
                   wait=0.5, max_iters=100):
        if opt == 'model':
            for quad_id, field in zip(quad_ids, fields):
                self.set_field(quad_id, field, opt)
        elif opt == 'live':
            # Define diff[i] as the difference between the desired field and
            # the live/book field for quad i. Check each quad and 
            # increase/decrease the field of quad i by max_change if 
            # abs(D[i]) > max_change. Move on if no quads changed, otherwise 
            # wait a moment so that the  machine doesn't trip and try again.
            stop, iters = False, 0
            while not stop and iters < max_iters:
                stop, iters = True, iters + 1
                for quad_id, field in zip(quad_ids, fields):
                    diff = field - self.get_field(quad_id, 'book')
                    if abs(diff) > max_change:
                        stop = False
                        if diff > 0:
                            self.set_field(quad_id, book + max_change, opt)
                        else:
                            self.set_field(quad_id, book - max_chage, opt)
                time.sleep(wait)
            # Now we can set the field like normal
            for quad_id, field in zip(quad_ids, fields): 
                self.set_field(quad_id, field, opt)
    
    def set_ref_ws_phases(self, mux, muy, beta_lims=(40, 40), verbose=0):
        """Set x and y phases at reference wire-scanner.

        Parameters
        ----------
        mux, muy : float
            The desired phase advances at the reference wire-scanner [rad].
        beta_lims : (xmax, ymax)
            Maximum beta functions to allow from s=0 to ws24.
        verbose : int
            If greater than zero, print a before/after summary.
        """        
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.beta_lims = beta_lims
                self.target_phases = [mux, muy]
                
            def score(self, trial, variables):
                fields = get_trial_vals(trial, variables)   
                self.controller.set_fields(self.controller.ind_quad_ids[:-5], 
                                                    fields, 'model')
                self.controller.track()
                calc_phases = self.controller.get_ref_ws_phases()
                cost = norm(subtract(calc_phases, self.target_phases))
                return cost + self.penalty_function()
            
            def penalty_function(self):
                max_betas = self.controller.get_max_betas() 
                penalty = 0.
                for max_beta, beta_lim in zip(max_betas, self.beta_lims):
                    penalty += step_func(max_beta - beta_lim)
                return penalty**2
            
        scorer = MyScorer(self)
        var_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B12', 'B13', 
                     'B14', 'B15', 'B16', 'B17', 'B18', 'B19']
        bounds = (self.ps_lb[:-5], self.ps_ub[:-5])
        init_fields = self.default_fields[:-5]    
        self.restore_default_optics()
        fields = minimize(scorer, init_fields, var_names, bounds)
        if verbose > 0:
            print '  Desired phases : {:.3f}, {:.3f}'.format(mux, muy)
            print '  Calc phases    : {:.3f}, {:.3f}'.format(*self.get_ref_ws_phases())
            print '  Max betas (Q03 - WS24): {:.3f}, {:.3f}'.format(*self.get_max_betas())
            print '  Betas at target: {:.3f}, {:.3f}'.format(*self.get_betas_at_target())
        
    def get_ref_ws_phases(self):
        """Return x and y phases (mod 2pi) at reference wire-scanner."""
        ws_state = self.trajectory.statesForElement(self.ref_ws_id)[0]
        ws_phases = self.adaptor.computeBetatronPhase(ws_state)
        return ws_phases.getx(), ws_phases.gety()
    
    def set_betas_at_target(self, betas, max_beta=100., verbose=0):
        """Vary quads after last wire-scanner to set betas at the target.
        
        Parameters
        ----------
        betas : (beta_x, beta_y)
            The desired beta functions at the target.
        max_beta : float
            Maximum beta function to allow between ws24 and the target.
        verbose : int
            If greater than zero, print a before/after summary.
        """
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.betas = betas
                self.max_beta = max_beta
                
            def score(self, trial, variables):
                fields = get_trial_vals(trial, variables)            
                self.controller.set_fields(self.controller.ind_quad_ids[-5:], 
                                                    fields, 'model')
                self.controller.track()
                residuals = subtract(self.betas, self.controller.get_betas_at_target())
                return norm(residuals) + self.penalty_function()
                
            def penalty_function(self):
                max_betas = self.controller.get_max_betas('RTBT_Diag:WS24', None)
                penalty = 0.
                for max_beta in max_betas:
                    penalty += step_func(max_beta - self.max_beta)            
                return penalty**2
            
        scorer = MyScorer(self)
        var_names = ['B26', 'B27', 'B28', 'B29', 'B30']
        init_fields = self.get_fields(self.ind_quad_ids[-5:], 'model')
        bounds = (self.ps_lb[-5:], self.ps_ub[-5:])
        fields = minimize(scorer, init_fields, var_names, bounds)
        self.set_fields(self.ind_quad_ids[-5:], fields, 'model')
        if verbose > 0:
            print '  Desired betas: {:.3f}, {:.3f}'.format(*betas)
            print '  Calc betas   : {:.3f}, {:.3f}'.format(*self.get_betas_at_target())
            
    def get_phases_for_scan(self, phase_coverage, nsteps_per_dim):
        """Create array of phases for scan. 
        
        Note that this will reset the model optics to their default settings.

        Parameters
        ----------
        phase_coverages : float
            Range of phase advances to cover in radians. The phases are
            centered on the default phase. It is a pain because OpenXAL 
            computes the phases mod 2pi. 
        scans_per_dim : int
            Number of phases to scan for each dimension (x, y). The total 
            number of scans will be 2 * `scans_per_dim`.
        """
        self.restore_default_optics()
        mux0, muy0 = self.get_ref_ws_phases()
        
        def _get_phases_for_scan_1d(dim='x'):
            phase = {'x': mux0, 'y': muy0}[dim]
            min_phase = put_angle_in_range(phase - 0.5 * phase_coverage)
            max_phase = put_angle_in_range(phase + 0.5 * phase_coverage)
            # Difference between and max phase is always <= 180 degrees
            abs_diff = abs(max_phase - min_phase)
            if abs_diff > math.pi:
                abs_diff = 2*math.pi - abs_diff
            # Return list of phases
            step = abs_diff / (nsteps_per_dim - 1)
            phases = [min_phase]
            for _ in range(nsteps_per_dim - 1):
                phase = put_angle_in_range(phases[-1] + step)
                phases.append(phase)
            return phases
        
        phases =[]
        for mux in _get_phases_for_scan_1d('x'):
            phases.append([mux, muy0])
        for muy in _get_phases_for_scan_1d('y'):
            phases.append([mux0, muy])
        return phases