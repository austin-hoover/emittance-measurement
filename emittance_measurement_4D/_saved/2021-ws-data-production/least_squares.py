"""
Solve linear least squares problem using Java matrices.
"""
from Jama import Matrix

A = Matrix([[ 0.,  1.],
            [ 1.,  1.],
            [ 2.,  1.],
            [ 3.,  1.]])
b = Matrix([[-1], [0.2], [0.9], [2.1]])
x = A.solve(b)
for i in range(2):
    print x.get(i, 0)