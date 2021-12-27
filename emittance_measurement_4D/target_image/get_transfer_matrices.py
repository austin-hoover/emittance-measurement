"""
This script loads the field strengths for each step in the scan. For 
each set of field strengths, the transfer matrices from each node to 
the target are calculated and saved to a file.
"""
from __future__ import print_function
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics

# Initialize the controller.
controller = optics.PhaseController(kinetic_energy=1.0e9, sync_mode='design', connect=False)

# Read the quadrupole field strengths.
file = open('_output/data/fields.dat', 'r')
lines = [line.rstrip() for line in file]
quad_ids = lines[0].split()
fields_list = [[float(s) for s in line.split()] for line in lines[1:]]
file.close()

# For each setting, extract the transfer matrix from each node to the target.
rec_node_id = 'RTBT:Tgt'
for i, fields in enumerate(fields_list):
    print('i = {}'.format(i))
    print('    Setting model optics.')
    controller.set_fields(quad_ids, fields, 'model')
    print('    Saving transfer matrices to file.')
    file = open('_output/data/tmats_{}.dat'.format(i), 'w')
    file.write('node_id M11 M12 M13 M14 M21 M22 M23 M34 M31 M32 M33 M34 M41 M42 M43 M44\n')
    for node in controller.sequence.getNodes():
        M = controller.transfer_matrix(node.getId(), rec_node_id)
        file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            node.getId(), *[M[i][j] for i in range(4) for j in range(4)])
        )
    file.close()
        
exit()