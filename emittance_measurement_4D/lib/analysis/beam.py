"""Compute parameters from covariance matrix."""
from math import sqrt, sin, cos, atan2
from jama import Matrix


def rms_ellipse_dims(Sigma, dim1='x', dim2='y'):
    str_to_int = {'x':0, 'xp':1, 'y':2, 'yp':3}
    i = str_to_int[dim1]
    j = str_to_int[dim2]
    sii, sjj, sij = Sigma.get(i, i), Sigma.get(j, j), Sigma.get(i, j)
    angle = -0.5 * atan2(2*sij, sii-sjj)
    sin, cos = sin(angle), cos(angle)
    sin2, cos2 = sin**2, cos**2
    c1 = sqrt(abs(sii*cos2 + sjj*sin2 - 2*sij*sin*cos))
    c2 = sqrt(abs(sii*sin2 + sjj*cos2 + 2*sij*sin*cos))
    return angle, c1, c2


def intrinsic_emittances(Sigma):
    U = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    SU = Sigma.times(U)
    SU2 = SU.times(SU)
    trSU2 = SU2.trace()
    detS = Sigma.det()
    eps_1 = 0.5 * sqrt(-trSU2 + sqrt(trSU2**2 - 16 * detS))
    eps_2 = 0.5 * sqrt(-trSU2 - sqrt(trSU2**2 - 16 * detS)) 
    return eps_1, eps_2


def apparent_emittances(Sigma):
    eps_x = sqrt(Sigma.get(0, 0) * Sigma.get(1, 1) - Sigma.get(0, 1)**2)
    eps_y = sqrt(Sigma.get(2, 2) * Sigma.get(3, 3) - Sigma.get(2, 3)**2)
    return eps_x, eps_y


def get_twiss2D(Sigma)
    eps_x, eps_y = apparent_emittances(Sigma)
    beta_x = Sigma[0][0] / eps_x
    beta_y = Sigma[2][2] / eps_y
    alpha_x = -Sigma[0][1] / eps_x
    alpha_y = -Sigma[2][3] / eps_y
    return np.array([alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y])