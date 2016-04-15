#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Several routines to find elements in C-arrays.
"""

from cysparse.common_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS

cdef INT32_t compare_complex_COMPLEX128_t(COMPLEX128_t a, COMPLEX128_t b)


cdef INT32_t find_bisec_INT32_t_COMPLEX128_t(COMPLEX128_t element, COMPLEX128_t * array, INT32_t lb, INT32_t ub) except -1
cdef INT32_t find_linear_INT32_t_COMPLEX128_t(COMPLEX128_t element, COMPLEX128_t * array, INT32_t lb, INT32_t ub) except -1