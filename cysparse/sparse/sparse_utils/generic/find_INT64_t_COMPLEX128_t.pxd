#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Several routines to find elements in C-arrays.
"""

from cysparse.common_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS

cdef INT32_t compare_complex_COMPLEX128_t(COMPLEX128_t a, COMPLEX128_t b)


cdef INT64_t find_bisec_INT64_t_COMPLEX128_t(COMPLEX128_t element, COMPLEX128_t * array, INT64_t lb, INT64_t ub) except -1
cdef INT64_t find_linear_INT64_t_COMPLEX128_t(COMPLEX128_t element, COMPLEX128_t * array, INT64_t lb, INT64_t ub) except -1