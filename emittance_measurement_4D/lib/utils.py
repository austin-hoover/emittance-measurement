import math
import os


def list_to_string(items):
    """Example: [1, 2, 3] -> '1 2 3'."""
    string = ''
    for item in items:
        string += '{} '.format(item)
    return string[:-1]


def shape(array):
    if type(array) != list:
        return []
    return [len(array)] + shape(array[0])


def linspace(start, stop, num=10):
    if num < 2:
        return [start]
    step = float(stop - start) / (num - 1)
    return [start + i*step for i in range(num)]


def arange(start, stop, step=1.0):
    vals, x = [], start
    while abs(x - stop) > abs(step):
        vals.append(x)
        x += step
    return vals


def multiply(vec, factor):
    return [factor * elem for elem in vec]


def add(*vecs):
    return [sum(elems) for elems in zip(*vecs)]


def subtract(vec1, vec2):
    return add(vec1, multiply(vec2, -1))


def norm(vec):
    return math.sqrt(sum([elem**2 for elem in vec]))


def dot(A, x):
    """Apply matrix `A` to vector `x`."""
    l, m, n = len(A), len(A[0]), len(x)
    if m != n:
        print('Shapes are incorrect.')
        return
    result = []
    for i in range(l):
        value = 0.
        for j in range(n):
            value += A[i][j] * x[j]
        result.append(value)
    return result


# Other functions
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
def delete_files_not_folders(directory):
    """Delete all files in directory and subdirectories."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith('.'):
                os.remove(os.path.join(root, file))
            
            
def save_array(array, filename):
    """Save list or list of lists to a file."""
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