from pprint import pprint
from Jama import Matrix

from lib.utils import load_array
from lib.analysis.reconstruct import reconstruct
from lib.analysis.least_squares import lsq_linear


A = load_array('A.dat')
b = load_array('b.dat')

moments = lsq_linear(A, b, verbose=2, max_iter=100, solver='lsmr')
print moments