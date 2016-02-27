#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Several routines to find elements in C-arrays.
"""

from cysparse.common_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS


cdef INT32_t find_bisec_INT32_t_INT32_t(INT32_t element, INT32_t * array, INT32_t lb, INT32_t ub) except -1
cdef INT32_t find_linear_INT32_t_INT32_t(INT32_t element, INT32_t * array, INT32_t lb, INT32_t ub) except -1