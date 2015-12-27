"""
Several routines to find elements in C-arrays.
"""

from cysparse.common_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS


cdef INT64_t find_bisec_INT64_t_INT32_t(INT32_t element, INT32_t * array, INT64_t lb, INT64_t ub) except -1
cdef INT64_t find_linear_INT64_t_INT32_t(INT32_t element, INT32_t * array, INT64_t lb, INT64_t ub) except -1