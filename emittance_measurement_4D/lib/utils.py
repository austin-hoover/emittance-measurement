"""General utility functions."""
import math
import os
from Jama import Matrix


def list_files(path, join=True):
    files = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and not file.startswith('.'):
            if join:
                files.append(file_path)
            else:
                files.append(file)
    return files

# Lists
#-------------------------------------------------------------------------------
def split(items, token):
    """Split `items` into sublists, excluding `token`.

    Example:
    >>> items = ['cat', 'dog', 'x', 'tree', 'bark']
    >>> split_list(items, 'x')
    [['cat', 'dog'], ['tree', 'bark']]
    """
    indices = [i for i, item in enumerate(items) if item == token]
    sublists = []
    if items[0] != token:
        sublists.append(items[:indices[0]])
    for lo, hi in zip(indices[:-1], indices[1:]):
        sublists.append(items[lo + 1:hi])
    if items[-1] != token:
        sublists.append(items[indices[-1] + 1:])
    return sublists


def list_to_string(items):
    """Example: [1, 2, 3] -> '1 2 3'."""
    string = ''
    for item in items:
        string += '{} '.format(item)
    return string[:-1]


def string_to_list(string):
    """Convert string to list of floats.
    
    '1 2 3' -> [1.0, 2.0, 3.0])
    """
    return [float(token) for token in string.split()]


def multiply(vec, factor):
    return [factor * elem for elem in vec]


def add(*vecs):
    return [sum(elems) for elems in zip(*vecs)]


def subtract(vec1, vec2):
    return add(vec1, multiply(vec2, -1))


def square(vec):
    return [elem**2 for elem in vec]


def norm(vec):
    return math.sqrt(sum(square(vec)))


def dot(vec1, vec2):
    return sum([elem1 * elem2 for elem1, elem2 in zip(vec1, vec2)])


def matvec(A, vec):
    return [dot(row, vec) for row in A]


def linspace(start, stop, num=10, endpoint=True):
    if num < 2:
        return [start]
    if endpoint:
        step = float(stop - start) / (num - 1)
    else:
        step = float(stop - start) / (num)
    return [start + i*step for i in range(num)]


def shape(array):
    if type(array) != list:
        return []
    return [len(array)] + shape(array[0])


def transpose(array):
    return [list(x) for x in zip(*array)]


def mean(xx):
    return sum(xx) / len(xx)


def variance(xx):
    x_avg = mean(xx)
    return sum([(x - x_avg)**2 for x in xx]) / len(xx)


def std(xx):
    return math.sqrt(variance(xx))


# JAMA matrices
#-------------------------------------------------------------------------------
def diagonal_matrix(diagonal_elements):
    n = len(diagonal_elements)
    M = Matrix(n, n)
    for i, elem in enumerate(diagonal_elements):
        M.set(i, i, elem)
    return M


def identity_matrix(n):
    return diagonal_matrix(n * [1.0])


def col_mat_to_list(M):
    """Convert Jama matrix of shape (n, 1) to list of shape (n,)."""
    return [M.get(i, 0) for i in range(M.getRowDimension())]


def list_to_col_mat(x):
    """Convert list of shape (n,) to Jama matrix of shape (n, 1)."""
    M = Matrix(len(x), 1)
    for i in range(len(x)):
        M.set(i, 0, x[i])
    return M


def get_col(M, col):
    """Get column of Jama matrix as list."""
    column = M.getMatrix(0, M.getRowDimension() - 1, [col])
    return col_mat_to_list(column)


def get_row(M, row):
    """Get row of Jama matrix as list."""
    row = M.getMatrix([row], 0, M.getColumnDimension())
    return list(row.getArray())

def mean_cols(M):
    if type(M) is list:
        M = Matrix(M)
    return [mean(get_col(M, j)) for j in range(M.getColumnDimension())]


def std_cols(M):
    if type(M) is list:
        M = Matrix(M)
    return [std(get_col(M, j)) for j in range(M.getColumnDimension())]


def cond(A):
    """Return condition number of square matrix A."""
    if type(A) is list:
        A = Matrix(A)
    return A.normF() * (A.inverse()).normF()



# Miscellaneous
#-------------------------------------------------------------------------------
def clip(x, lo=None, hi=None):
    "Enforce lo <= x <= hi."
    if lo is not None and x < lo:
        x = lo
    elif hi is not None and x > hi:
        x = hi
    return x


def radians(angle_in_degrees):
    """Convert degrees to radians."""
    return angle_in_degrees * (math.pi / 180)


def degrees(angle_in_radians):
    """Convert radians to degrees."""
    return angle_in_radians * (180 / math.pi)
    
    
def put_angle_in_range(angle):
    """Put angle in range [0, 2pi]."""
    lim = 2 * math.pi
    if angle < 0:
        if abs(angle) > lim:
            angle = -(abs(angle) % lim)
        angle += lim
    if angle > lim:
        angle %= lim
    return angle


# File processing
#-------------------------------------------------------------------------------
def delete_files_not_folders(directory):
    """Delete all files in directory and subdirectories."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith('.'):
                os.remove(os.path.join(root, file))
            
            
def save_array(array, filename):
    if len(shape(array)) == 1:
        array = [array]
        
    n_rows = len(array)
    file = open(filename, 'w')
    for i, row in enumerate(array):
        string = list_to_string(row)
        if i < n_rows - 1:
            string += '\n'
        file.write(string)
    file.close()
    
    
def load_array(filename):
    file = open(filename, 'r')
    array = []
    for line in file:
        line = line.rstrip()
        array.append([float(token) for token in line.split()])
    file.close()
    return array