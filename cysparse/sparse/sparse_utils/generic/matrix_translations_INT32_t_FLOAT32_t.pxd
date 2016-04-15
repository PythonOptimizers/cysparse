#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Diverse utilities to translate one matrix format into another.

"""
from cysparse.common_types.cysparse_types cimport *

cdef csr_to_csc_kernel_INT32_t_FLOAT32_t(INT32_t nrow, INT32_t ncol, INT32_t nnz,
                                      INT32_t * csr_ind, INT32_t * csr_col, FLOAT32_t * csr_val,
                                      INT32_t * csc_ind, INT32_t * csc_row, FLOAT32_t * csc_val)

cdef csc_to_csr_kernel_INT32_t_FLOAT32_t(INT32_t nrow, INT32_t ncol, INT32_t nnz,
                                      INT32_t * csc_ind, INT32_t * csc_row, FLOAT32_t * csc_val,
                                      INT32_t * csr_ind, INT32_t * csr_col, FLOAT32_t * csr_val)

cdef csc_to_ll_kernel_INT32_t_FLOAT32_t(INT32_t nrow, INT32_t ncol, INT32_t nnz,
                                      INT32_t * csc_ind, INT32_t * csc_row, FLOAT32_t * csc_val,
                                      INT32_t * ll_root, INT32_t * ll_col, INT32_t * ll_link, FLOAT32_t * ll_val)