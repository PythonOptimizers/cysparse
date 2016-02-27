#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Several routines to find elements in C-arrays.
"""

from cysparse.common_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS

cdef INT32_t compare_complex_COMPLEX64_t(COMPLEX64_t a, COMPLEX64_t b)


cdef INT32_t find_bisec_INT32_t_COMPLEX64_t(COMPLEX64_t element, COMPLEX64_t * array, INT32_t lb, INT32_t ub) except -1
cdef INT32_t find_linear_INT32_t_COMPLEX64_t(COMPLEX64_t element, COMPLEX64_t * array, INT32_t lb, INT32_t ub) except -1