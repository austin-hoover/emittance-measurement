"""Scan phase advances at WS24; save magnet strengths."""
from __future__ import print_function
import sys
import os
from pprint import pprint

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
from xal.smf.impl import MagnetMainSupply
from xal.tools.beam import CovarianceMatrix
from xal.tools.beam import PhaseVector
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib import utils


controller = optics.PhaseController(sync_mode='design', connect=False)

phase_coverage = 179.0 # [deg]
n_steps = 15
max_beta = 35.0 # [m/rad]
beta_lims = (max_beta, max_beta)


# Get list of phases for x and y (low to high)
phases = controller.get_phases_for_scan(phase_coverage, n_steps, method=1)
muxx = [mux for (mux, muy) in phases]
muyy = [muy for (mux, muy) in phases]
muyy = list(reversed(muyy))

# These are the quads that will will be varied.
quad_ids = ['RTBT_Mag:QH18', 'RTBT_Mag:QV19', 'RTBT_Mag:QH26', 'RTBT_Mag:QV27', 
            'RTBT_Mag:QH28', 'RTBT_Mag:QV29', 'RTBT_Mag:QH30']


phases_at_tgt = []
phases_at_ws24 = []


file = open('_output/optics.dat', 'w')
for quad_id in quad_ids:
    file.write(quad_id + ' ')
file.write('\n')

for i, mux in enumerate(muxx): 
    for j, muy in enumerate(muyy):
        
        print('i, j = {}, {}'.format(i, j))
        
        # Save phase advances at WS24.
        mux_deg = utils.degrees(mux)
        muy_deg = utils.degrees(muy)
        phases_at_ws24.append([mux_deg, muy_deg])

        # Set model phase advances.
        controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)

        # Constrain beta functions on target if too far from default.
        beta_x_target, beta_y_target = controller.beta_funcs('RTBT:Tgt')
        beta_x_default, beta_y_default = controller.default_betas_at_target
        frac_change_x = abs(beta_x_target - beta_x_default) / beta_x_default
        frac_change_y = abs(beta_y_target - beta_y_default) / beta_y_default
        tol = 0.05
        if frac_change_x > tol or frac_change_y > tol:
            print('  Setting betas at target...')
            controller.constrain_size_on_target(verbose=0)
        max_betas_anywhere = controller.max_betas(stop=None)
        print('  Max betas anywhere: {:.3f}, {:.3f}.'.format(*max_betas_anywhere))

        # Save phase advances at WS24.
        mux_tgt, muy_tgt = controller.phases('RTBT:Tgt')
        mux_tgt = utils.degrees(mux_tgt)
        muy_tgt = utils.degrees(muy_tgt)
        phases_at_tgt.append([mux_tgt, muy_tgt])
        print('  Phase advances at WS24   = {:.2f}, {:.2f} [deg]'.format(mux_deg, muy_deg))
        print('  Phase advances at target = {:.2f}, {:.2f} [deg]'.format(mux_tgt, muy_tgt))
        print()

        # Write quad fields to a file.
        fields = controller.get_fields(quad_ids, 'model')
        for field in fields:
            file.write('{} '.format(field))
        file.write('\n')
        
file.close()
    
    
# Save phase advances to file.
file = open('_output/phases_at_tgt.dat', 'w')
for (mux, muy) in phases_at_tgt:
    file.write('{} {}\n'.format(mux, muy))
file.close()

file = open('_output/phases_at_ws24.dat', 'w')
for (mux, muy) in phases_at_ws24:
    file.write('{} {}\n'.format(mux, muy))
file.close()

exit()