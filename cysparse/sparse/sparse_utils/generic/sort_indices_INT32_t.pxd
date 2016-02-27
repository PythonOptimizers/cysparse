#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.common_types.cysparse_types cimport *


cdef sort_array_INT32_t(INT32_t * a, INT32_t start, INT32_t end)