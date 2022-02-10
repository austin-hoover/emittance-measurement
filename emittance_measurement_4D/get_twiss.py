"""Save the Twiss parameters at each node in the RTBT."""
from __future__ import print_function
import sys
import os
from lib import optics


pvloggerid = 49547664
kinetic_energy = 0.8e9  # [eV]

controller = optics.PhaseController(kinetic_energy=kinetic_energy)
controller.sync_model_pvloggerid(pvloggerid)

file = open("_output/data/model_twiss.dat", "w")
file.write("node_id alpha_x alpha_y beta_x beta_y\n")
for node in controller.sequence.getNodes():
    print(node.getId())
    mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y = controller.twiss(
        node.getId()
    )
    file.write(
        "{} {} {} {} {}\n".format(node.getId(), alpha_x, alpha_y, beta_x, beta_y)
    )
file.close()

exit()
