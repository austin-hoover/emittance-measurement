"""Compute the transfer matrices from each node to each wire-scanner."""
from __future__ import print_function
import sys
import os
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib import optics


pvloggerid = 49547664
kinetic_energy = 0.8e9
accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
tmatgen = optics.TransferMatrixGenerator(sequence, kinetic_energy)
tmatgen.sync(pvloggerid)

ws_ids = ['RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
for ws_id in ws_ids:
    file = open('_output/data/tmats_{}.dat'.format(ws_id), 'w')
    file.write('node_id M11 M12 M13 M14 M21 M22 M23 M34 M31 M32 M33 M34 M41 M42 M43 M44\n')
    for node in sequence.getNodes():
        node_id = node.getId()
        print(node_id)
        M = tmatgen.generate(node_id, ws_id)
        file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            node.getId(), *[M[i][j] for i in range(4) for j in range(4)])
        )
    file.close()
    
    file = open('_output/data/node_positions.dat', 'w')
    first_node = sequence.getNodes()[0]
    for node in sequence.getNodes():
        position = sequence.getDistanceBetween(first_node, node)
        file.write('{} {}\n'.format(node.getId(), position))
    file.close()
    
exit()