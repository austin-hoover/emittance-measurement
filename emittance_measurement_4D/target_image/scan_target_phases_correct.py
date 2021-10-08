"""Try to gain better control of the phase advances at the target."""
from __future__ import print_function
import sys
import os
import time

from xal.extension.solver import Scorer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics
from lib import utils
from lib.utils import clip
from lib.utils import norm
from lib.utils import subtract
from lib.utils import radians, degrees
from lib.xal_helpers import get_trial_vals
from lib.xal_helpers import minimize
from lib.xal_helpers import write_traj_to_file


def set_target_phases(mux, muy, beta_max_before_ws24, beta_max_after_ws24,
                      default_target_betas, target_beta_frac_tol, controller):
    """Set x and y phases at the target.

    Parameters
    ----------
    mux, muy : float
        The desired phase advances at the target [rad].
    beta_max_before_ws24 : float
        Maximum beta function to allow before WS24.
    beta_max_after_ws24 : float
        Maximum beta function to allow after WS24.
    default_target_betas : (beta_x, beta_y)
        The default beta functions at the target.
    target_beta_frac_tol : float
        Fractional tolerance for target beta functions.
    controller : PhaseController
        RTBT optics controller. Eventually this method will become a PhaseController method.
    """        
    class MyScorer(Scorer):
        def __init__(self, controller, quad_ids):
            self.controller = controller
            self.quad_ids = quad_ids
            self.target_phases = [mux, muy]

        def score(self, trial, variables):
            fields = get_trial_vals(trial, variables)   
            self.controller.set_fields(self.quad_ids, fields, 'model')
            self.controller.track()
            calc_phases = self.controller.phases('RTBT:Tgt')
            residuals = subtract(calc_phases, self.target_phases)
            cost = norm(residuals)**2
            cost += self.penalty_max_beta()
            cost += self.penalty_target_betas()
            # print('  cost = {}'.format(cost))
            return cost

        def penalty_max_beta(self):
            penalty = 0.0
            for beta in self.controller.max_betas(start='RTBT_Mag:QH18', stop='RTBT_Diag:WS24'):
                penalty += clip(beta - beta_max_before_ws24, 0.0, None)**2
            for beta in self.controller.max_betas(start='RTBT_Diag:WS24', stop='RTBT:Tgt'):
                penalty += clip(beta - beta_max_after_ws24, 0.0, None)**2
            return penalty

        def penalty_target_betas(self):
            penalty = 0.0
            target_betas = self.controller.beta_funcs('RTBT:Tgt')
            diffs = subtract(target_betas, default_target_betas)
            diff_x, diff_y = diffs
            abs_frac_change_x = abs(diff_x / default_target_betas[0])
            abs_frac_change_y = abs(diff_y / default_target_betas[1])
            if abs_frac_change_x > target_beta_frac_tol:
                penalty += diff_x**2
            if abs_frac_change_y > target_beta_frac_tol:
                penalty += diff_y**2
            return penalty

    lo = controller.ind_quad_ids.index('RTBT_Mag:QH18')
    hi = controller.ind_quad_ids.index('RTBT_Mag:QH30')
    quad_ids = var_names = controller.ind_quad_ids[lo: hi + 1]
    scorer = MyScorer(controller, quad_ids)
    lb = controller.ps_lb[lo: hi + 1]
    ub = controller.ps_ub[lo: hi + 1]
    bounds = (lb, ub)
    guess = controller.get_fields(quad_ids, 'model')
    minimize(scorer, guess, var_names, bounds)


quad_ids = ['RTBT_Mag:QH18', 'RTBT_Mag:QV19', 'RTBT_Mag:QH26', 'RTBT_Mag:QV27',
            'RTBT_Mag:QH28', 'RTBT_Mag:QV29', 'RTBT_Mag:QH30']
kinetic_energy = 1.0e9

controller = optics.PhaseController(kinetic_energy=kinetic_energy)
mux0, muy0 = controller.phases('RTBT:Tgt')
default_target_betas = controller.beta_funcs('RTBT:Tgt')
default_fields = controller.get_fields(quad_ids, 'model')
print('Initial (default):')
print('  Phase advances at target = {:.2f}, {:.2f}'.format(mux0, muy0))
print('  Max betas anywhere (< WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(stop='RTBT_Diag:WS24')))
print('  Max betas anywhere (> WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(start='RTBT_Diag:WS24', stop='RTBT:Tgt')))
print('  Betas at target = {:.2f}, {:.2f}'.format(*controller.beta_funcs('RTBT:Tgt')))


n_steps = 15
dmux_min = dmuy_min = radians(-55.)
dmux_max = dmuy_max = radians(124.)
beta_max_before_ws24 = 35.0
beta_max_after_ws24 = 95.0
target_beta_frac_tol = 0.15

muxx = optics.lin_phase_range(mux0 + dmux_min, mux0 + dmux_max, n_steps)
muyy = optics.lin_phase_range(muy0 + dmuy_min, muy0 + dmuy_max, n_steps)


file_phase_adv = open('_output/phase_adv.dat', 'w')
file_fields = open('_output/fields.dat', 'w')
file_default_fields = open('_output/default_fields.dat', 'w')

for quad_id in quad_ids:
    file_fields.write(quad_id + ' ')
    file_default_fields.write(quad_id + ' ')
file_fields.write('\n')
file_default_fields.write('\n')


start_time = time.time()

counter = 0
for i, mux in enumerate(muxx):
    for j, muy in enumerate(muyy):

        print('i, j, time = {}, {}, {}'.format(i, j, time.time() - start_time))

        set_target_phases(mux, muy, beta_max_before_ws24, beta_max_after_ws24,
                          default_target_betas, target_beta_frac_tol, controller)

        mux_calc, muy_calc = controller.phases('RTBT:Tgt')
        fields = controller.get_fields(quad_ids, 'model')

        # print('Final:')
        # print('  Phase advances at target = {:.2f}, {:.2f} [deg]'.format(degrees(mux_calc), degrees(muy_calc)))
        # print('  Expected                 = {:.2f}, {:.2f} [deg]'.format(degrees(mux), degrees(muy)))
        # print('  Max betas anywhere (< WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(stop='RTBT_Diag:WS24')))
        # print('  Max betas anywhere (> WS24) = {:.2f}, {:.2f}'.format(*controller.max_betas(start='RTBT_Diag:WS24', stop='RTBT:Tgt')))
        # print('  Betas at target = {:.2f}, {:.2f}'.format(*controller.beta_funcs('RTBT:Tgt')))
        # print('Magnet changes (id, new, old, abs(frac_change):')
        # for quad_id, field, default_field in zip(quad_ids, fields, default_fields):
        #     frac_change = abs(field - default_field) / default_field
        #     print('  {}: {} {} {}'.format(quad_id, field, default_field, frac_change))

        file_phase_adv.write('{} {}\n'.format(mux_calc, muy_calc))

        for field, default_field in zip(fields, default_fields):
            file_fields.write('{} '.format(field))
            file_default_fields.write('{} '.format(default_field))
        file_fields.write('\n')
        file_default_fields.write('\n')

        write_traj_to_file(controller.tracked_twiss(), controller.positions, '_output/twiss_{}.dat'.format(counter))
        counter += 1


print('Runtime = {}'.format(time.time() - start_time))

file_phase_adv.close()
file_fields.close()
file_default_fields.close()

exit()