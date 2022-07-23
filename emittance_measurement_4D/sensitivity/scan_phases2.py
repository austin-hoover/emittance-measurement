"""Scan the horizontal and vertical phase advances at WS24. 

At each (mux, muy) in the scan:
    1. Calculate the condition numbers of the coefficient matrices Axx, Ayy, Axy.
    2. Run a Monte Carlo simulation and record the fail rate + covariance matrix
       ensemble.
"""
from __future__ import print_function
import sys
import os
import math
import random
from Jama import Matrix

from xal.ca import Channel
from xal.ca import ChannelFactory
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq
from xal.smf.data import XMLDataManager
from xal.smf.impl import MagnetMainSupply
from xal.tools.beam import CovarianceMatrix
from xal.tools.beam import PhaseVector
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings

from helpers import get_moments
from helpers import run_trials2
from helpers import solve
from helpers import matched_cov

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib import analysis
from lib import optics
from lib.least_squares import lsq_linear
from lib import utils


utils.delete_files_not_folders("_output/")


# Setup
kinetic_energy = 1.0e9  # [eV]
ws_ids = ["RTBT_Diag:WS20", "RTBT_Diag:WS21", "RTBT_Diag:WS23", "RTBT_Diag:WS24"]
ref_ws_id = "RTBT_Diag:WS24"
rec_node_id = "RTBT_Diag:BPM17"
n_steps_x = n_steps_y = 12
dmux_lo = utils.radians(-45.0)
dmux_hi = utils.radians(+45.0)
dmuy_lo = utils.radians(-45.0)
dmuy_hi = utils.radians(+45.0)
n_trials = 1000
frac_error = 0.03
controller = optics.PhaseController(ref_ws_id=ref_ws_id, kinetic_energy=kinetic_energy)

pvloggerid = 48842900  # 2021/08/01, production optics
if pvloggerid is not None:
    controller.sync_model_pvloggerid(pvloggerid)

print("Betas at target:", controller.beta_funcs("RTBT:Tgt"))
print("Phase advances at target:", controller.phases(ref_ws_id))

mux0, muy0 = controller.phases(ref_ws_id)
quad_ids = controller.ind_quad_ids
default_fields = controller.get_fields(quad_ids)

# Create initial matched covariance matrix (uncoupled).
eps_x = 20.0e-6  # [m rad]
eps_y = 20.0e-6  # [m rad]
rec_node_id = "RTBT_Diag:BPM17"
_, _, alpha_x, alpha_y, beta_x, beta_y, _, _ = controller.twiss(rec_node_id)
print("Model Twiss parameters:")
print(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y)

Sigma0 = matched_cov(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y, c=0.5)

print("Beam Twiss parameters:")
stats = analysis.BeamStats(Sigma0)
print(
    stats.alpha_x,
    stats.alpha_y,
    stats.beta_x,
    stats.beta_y,
    stats.eps_x,
    stats.eps_y,
    stats.eps_1,
    stats.eps_2,
)
file = open("_output/data/true_emittances.dat", "w")
for eps in [stats.eps_x, stats.eps_y, stats.eps_1, stats.eps_2]:
    file.write("{} ".format(eps))
file.close()


# Compute condition number for Axy over grid of x and y phase advances.
# ------------------------------------------------------------------------------
def cond(A):
    if A.getRowDimension() == A.getColumnDimension():
        Ainv = A.inverse()
    else:
        AT = A.transpose()
        Ainv = ((AT.times(A)).inverse()).times(AT)
    return A.normF() * Ainv.normF()


dmuxx = utils.linspace(dmux_lo, dmux_hi, n_steps_x)
dmuyy = utils.linspace(dmuy_lo, dmuy_hi, n_steps_y)
    
fail_rates = Matrix(n_steps_x, n_steps_y)
Sigmas = []

print()
print(
    "i/N  | j/N |fail rate"
)
print(
    "-------------------------------------------------------------------------------------------------------"
)
for i, dmux in enumerate(dmuxx):
    Sigmas.append([])
    mux = utils.put_angle_in_range(mux0 + dmux)
    for j, dmuy in enumerate(dmuyy):
        muy = utils.put_angle_in_range(muy0 + dmuy)
        controller.set_ref_ws_phases(mux, muy, verbose=1)
        _mux, _muy = controller.phases(ref_ws_id)
        tmats = [controller.transfer_matrix(rec_node_id, ws_id) for ws_id in ws_ids]
        Axx, Ayy, Axy = [], [], []
        for M in tmats:
            Axx.append([M[0][0] ** 2, M[0][1] ** 2, 2 * M[0][0] * M[0][1]])
            Ayy.append([M[2][2] ** 2, M[2][3] ** 2, 2 * M[2][2] * M[2][3]])
            Axy.append(
                [
                    M[0][0] * M[2][2],
                    M[0][1] * M[2][2],
                    M[0][0] * M[2][3],
                    M[0][1] * M[2][3],
                ]
            )
        Axx = Matrix(Axx)
        Ayy = Matrix(Ayy)
        Axy = Matrix(Axy)
        A = Matrix(10, 10)
        for k in range(3):
            for l in range(3):
                A.set(k, l, Axx.get(k, l))
                A.set(k + 3, l + 3, Ayy.get(k, l))
        for k in range(4):
            for l in range(4):
                A.set(k + 6, l + 6, Axy.get(k, l))
        condition_number = cond(A)
        condition_number_xx = cond(Axx)
        condition_number_yy = cond(Ayy)
        condition_number_xy = cond(Axy)

        fail_rate, _Sigmas = run_trials2(Sigma0, tmats, n_trials, frac_error)
        fail_rates.set(i, j, fail_rate)
        Sigmas[i].append(_Sigmas)
        print(
            "{}/{} | {}/{} | {:.2f}".format(
                i, n_steps_x - 1,
                j, n_steps_y - 1,
                fail_rate,
            )
        )
        controller.set_fields(quad_ids, default_fields, "model")

utils.save_pickle("_output/data/Sigmas.pkl", Sigmas)
utils.save_array(fail_rates.getArray(), "_output/data/fail_rates.dat")
utils.save_array(dmuxx, "_output/data/phase_devs_x.dat")
utils.save_array(dmuyy, "_output/data/phase_devs_y.dat")
exit()
