#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.common_types.cysparse_types cimport *


cdef sort_array_INT64_t(INT64_t * a, INT64_t start, INT64_t end)