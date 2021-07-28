"""Test reconstruction of covariance matrix.

The exact least-squares solution seems to give a beam that has nonzero
cross-plane correlations. In scipy, I can switch the method to 'lsmr'
and it works better. We can also use nonlinear least squares or any 
other optimizer. 

I'm not sure why this is. The x-x' and y-y' projections are fit well.

This script just demonstrates that the exact llsq solution can be 
found in OpenXAL (without NumPy). There is no reason that we can't
brute-force it.

I'm considering adding a tab to the GUI that will be able to load PTA
files, compute the 4D parameters, and plot the results. Perhaps we
can load the MachineSnapshot from the PVLoggerID as in C. Allen's app. 
Then it will do the following after the files are uploaded:
    * Parse the files.
    * Compute all needed transfer matrices.
    * Build observation and measurement arrays.
    * Solve least-squares problem.
    * Plot the result.
    * Dump all data in a nice format for later use.
    
The reason for considering this is because C. Allen's app can only 
use the fixed-optics approach, and it's in Java so it might be hard to 
edit. 

So the workflow would be:
    1. Make sure all five wire-scans are activated, then take series of 
       wire-scans at different optics using the PTA app. (The optics can 
       be changed using my app, if desired.) At each step, save the PTA file. 
       I think the filename can be left as is since it will be unique. 
    2. Load all the PTA files into my app.
    3. Within app: Solve the problem.
    4. Within app: Display the results.
"""

from math import sqrt
from Jama import Matrix

def load_matrix(filename):
    file = open(filename, 'r')
    M = []
    for line in file:
        line = line.rstrip()
        M.append([float(token) for token in line.split()])
    file.close()
    return Matrix(M)

A = load_matrix('A.dat')
b = load_matrix('b.dat')
moment_vec = A.solve(b)

sig_11, sig_22, sig_12, sig_33, sig_44, sig_34, sig_13, sig_23, sig_14, sig_24 = [moment_vec.get(i, 0) for i in range(10)]
Sigma = Matrix([[sig_11, sig_12, sig_13, sig_14], 
                [sig_12, sig_22, sig_23, sig_24],
                [sig_13, sig_23, sig_33, sig_34],
                [sig_14, sig_24, sig_34, sig_44]])

for i in range(4):
    string = ''
    for j in range(4):
        string += '{} '.format(Sigma.get(i, j))
    print string
    
eps_4D = sqrt(Sigma.det())
eps_x = sqrt(Sigma.get(0, 0) * Sigma.get(1, 1) - Sigma.get(0, 1)**2)
eps_y = sqrt(Sigma.get(2, 2) * Sigma.get(3, 3) - Sigma.get(2, 3)**2)

U = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
SU = Sigma.times(U)
SU2 = SU.times(SU)
trSU2 = SU2.trace()
detS = Sigma.det()
eps_1 = 0.5 * sqrt(-trSU2 + sqrt(trSU2**2 - 16 * detS))
eps_2 = 0.5 * sqrt(-trSU2 - sqrt(trSU2**2 - 16 * detS))    

print 'eps_x = {} [mm mrad]'.format(eps_x)
print 'eps_y = {} [mm mrad]'.format(eps_y)
print 'eps_1 = {} [mm mrad]'.format(eps_1)
print 'eps_2 = {} [mm mrad]'.format(eps_2)
print 'eps_4D = {} [mm^2 mrad^2]'.format(eps_4D)
print 'eps1 * eps_2 = {} [mm^2 mrad^2]'.format(eps_1 * eps_2)