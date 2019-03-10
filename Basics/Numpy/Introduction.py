import numpy as np

## Attributes:
## dimensions of an array put into a tuple:
## n rows & m columns, length of np.shape is np.ndim.
## np.size, total number of arrays
## np.dtype, type of elements in an array
## np.itemsize, the size in bytes of each element in the array
## np.data, buffer containing elements of the array:
## not really needed, since we are accessing them with indices
## np.sum, adds up all the elements in the array
## np.sum(axis = 0), sum of elements over all columns
## np.sum(axis = 1), sum of elements over all rows

# one dimensional arrays
a = np.array([1, 5, 7])

# two dimensional arrays
b = np.array([(1, 5, 2, 3), (7, 8, 9)])

# arrays of zeros
zeroArray = np.zeros((3, 4))

# arrays of ones
onesArray = np.ones((3, 4))

# creates an array that stores "start, step, and stop" values
# - "stop" value is not included
np.arange(10, 40, 5)

# array of evenly spaced values
np.linspace(0, 2, 10)

# dimensions/axes an array has
dimensions = a.ndim, b.ndim

# returns a flattened array
ravelArray = a.ravel()

# modifies shape
modifiedArrayShape = a.reshape(6, 2)






