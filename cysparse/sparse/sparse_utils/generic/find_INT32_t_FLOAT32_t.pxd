#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Several routines to find elements in C-arrays.
"""

from cysparse.common_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS


cdef INT32_t find_bisec_INT32_t_FLOAT32_t(FLOAT32_t element, FLOAT32_t * array, INT32_t lb, INT32_t ub) except -1
cdef INT32_t find_linear_INT32_t_FLOAT32_t(FLOAT32_t element, FLOAT32_t * array, INT32_t lb, INT32_t ub) except -1