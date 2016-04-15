#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Several routines to find elements in C-arrays.
"""

from cysparse.common_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS


cdef INT64_t find_bisec_INT64_t_FLOAT64_t(FLOAT64_t element, FLOAT64_t * array, INT64_t lb, INT64_t ub) except -1
cdef INT64_t find_linear_INT64_t_FLOAT64_t(FLOAT64_t element, FLOAT64_t * array, INT64_t lb, INT64_t ub) except -1