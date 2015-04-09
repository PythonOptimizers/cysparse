"""
ll_vec extension.

Implements a dense (normal) vector type.
"""

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc,free

cnp.import_array()


cdef class DVector:


    def __cinit__(self, int n):
        self.n = n

    def __dealloc__(self):
        free(self.data)

    def __setitem__(self, int key, double value):

        cdef int i = key
        self.data[i] = value

    def __getitem__(self, int key):
        return self.data[key]

cdef class IVector:

    def __cinit__(self, int n):
        self.n = n

    def __dealloc__(self):
        free(self.data)

    def __setitem__(self, int key, int value):

        cdef int i = key
        self.data[i] = value

    def __getitem__(self, int key):
        cdef int val = self.data[key]
        print "gogo"
        print val
        return val

########################################################################################################################
# With numpy
########################################################################################################################

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        """ Set the data of the array

        This cannot be done in the constructor as it must recieve C-level
        arguments.

        Args:
            size (int): Length of the array.
            data_ptr (void*) Pointer to the data.

        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_INT, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.data_ptr)