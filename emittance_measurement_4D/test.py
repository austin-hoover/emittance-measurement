from __future__ import print_function
from pprint import pprint
from Jama import Matrix

Sigma = Matrix([[198.12072661438378, 43.24666078269238, -8.00926078382594, -0.6616349602640319],
                [43.24666078269238, 14.88814111485498, -0.8101365051180459, -0.11083651460686837],
                [-8.00926078382594, -0.8101365051180459, 268.6764396057542, -17.413922776216104], 
                [-0.6616349602640319, -0.11083651460686837, -17.413922776216104, 3.619663474439786]])
U = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
SigmaU = Sigma.times(U)

eig = SigmaU.eig()
V = eig.getV()
D = eig.getD()

imag_eigvals = eig.getImagEigenvalues()
intrinsic_emittances = [e for e in imag_eigvals]

print(intrinsic_emittances)
print()
print(D.getArray())
print()
print(V.getArray())


print()
print(SigmaU.getArray())
print()
print(V.times(D.times(V.inverse())).getArray())

Vinv = V.inverse()
Sigma_n = Vinv.times(Sigma.times(Vinv.transpose()))
print()
for row in Sigma_n.getArray():
    print(row)


# print()
# def get_col_if_contains(val):
#     for col in D.transpose().getArray():
#         print(col)
#         if val in col:
#             return val
#     return None

# print(get_col_if_contains(V, intrinsic_emittances[0]))