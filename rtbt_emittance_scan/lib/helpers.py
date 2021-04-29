"""Helper functions for OpenXAL scripts."""

import math
import random
from xal.smf import Accelerator
from xal.smf.data import XMLDataManager
from xal.extension.solver import Trial, Variable, Scorer, Stopper, Solver, Problem
from xal.extension.solver.ProblemFactory import getInverseSquareMinimizerProblem
from xal.extension.solver.SolveStopperFactory import maxEvaluationsStopper
from xal.extension.solver.algorithm import SimplexSearchAlgorithm

from utils import subtract, norm, step_func, dot, put_angle_in_range


def loadRTBT():
    """Load the RTBT sequence of the SNS accelerator."""
    accelerator = XMLDataManager.loadDefaultAccelerator()
    return accelerator.getComboSequence('RTBT')


def write_traj_to_file(data, positions, filename):
    """Save trajectory data to file. 
    `data[i]` is list of data at position `positions[i]`."""
    file = open(filename, 'w')
    fstr = len(data[0]) * '{} ' + '{}\n'
    for s, dat in zip(positions, data):
        file.write(fstr.format(s, *dat))
    file.close()
    

# Helper functions for OpenXAL Solver
def get_trial_vals(trial, variables):
    """Get list of variable values from Trial."""
    trial_point = trial.getTrialPoint()
    return [trial_point.getValue(var) for var in variables]


# def minimize(variables, scorer, solver, tol=1e-8):
#     """Run the solver to minimize the score."""
#     problem = getInverseSquareMinimizerProblem(variables, scorer, tol)
#     solver.solve(problem)
#     trial = solver.getScoreBoard().getBestSolution()
#     return get_trial_vals(trial, variables)

def minimize(scorer, x, var_names, bounds, maxiters=1000, tol=1e-8):
    """Minimize a multivariate function using the simplex algorithm.
    
    To do: ensure initial guess is within bounds.
    
    Parameters
    ----------
    scorer : Scorer subclass
        Must implement method `score(trial, variables).`
    x : list[float], shape (n,)
        Initial guess.
    var_names : list[str], shape(n,)
        List of variable names.
    bounds : 2-tuple of list
        Lower and upper bounds on independent variables. Each array must match
        the size of x or be a scalar; in the latter case the bound will be the 
        same for all variables.
    """ 
    n = len(x)
    if n != len(var_names):
        raise ValueError('Parameter list and variable name list have different length.')
    
    lb, ub = bounds
    if type(lb) in [float, int]:
        lb = n * [lb]
    if type(ub) in [float, int]:
        ub = n * [ub]
    for i in range(n):
        if x[i] < lb[i] or x[i] > ub[i]:
            raise ValueError('Initial guess is outside bounds.')
            
    variables = [Variable(name, val, l, u) for name, val, l, u in zip(var_names, x, lb, ub)]
    stopper = maxEvaluationsStopper(maxiters)
    solver = Solver(SimplexSearchAlgorithm(), stopper)
    problem = getInverseSquareMinimizerProblem(variables, scorer, tol)
    solver.solve(problem)
    trial = solver.getScoreBoard().getBestSolution()
    return get_trial_vals(trial, variables)
    

def least_squares(A, b, x=None, bounds=None, verbose=0):
    """Return the least-squares solution to the equation A.x = b.
    
    This will be used if we want to reconstruct the beam emittances from 
    within the app.
    """ 
    class MyScorer(Scorer):
        def __init__(self, A, b):
            self.A, self.b = A, b
        def score(self, trial, variables):
            x = get_trial_vals(trial, variables)
            residuals = subtract(dot(A, x), b)
            return norm(residuals)
    
    if bounds is None:
        bounds = (-float('inf'), float('inf'))
    n = len(A[0])
    var_names = ['v{}'.format(i) for i in range(n)]
    x = [random.random() for _ in range(n)] if x is None else x
    scorer = MyScorer(A, b)
    return minimize(scorer, x, var_names, bounds)
