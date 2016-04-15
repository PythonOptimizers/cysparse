#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
"""
Diverse utilities to translate one matrix format into another.

"""
from cysparse.common_types.cysparse_types cimport *

cdef csr_to_csc_kernel_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csr_ind, INT64_t * csr_col, INT32_t * csr_val,
                                      INT64_t * csc_ind, INT64_t * csc_row, INT32_t * csc_val)

cdef csc_to_csr_kernel_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csc_ind, INT64_t * csc_row, INT32_t * csc_val,
                                      INT64_t * csr_ind, INT64_t * csr_col, INT32_t * csr_val)

cdef csr_to_ll_kernel_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csr_ind, INT64_t * csr_col, INT32_t * csr_val,
                                      INT64_t * ll_root, INT64_t * ll_col, INT64_t * ll_link, INT32_t * ll_val)

cdef csc_to_ll_kernel_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csc_ind, INT64_t * csc_row, INT32_t * csc_val,
                                      INT64_t * ll_root, INT64_t * ll_col, INT64_t * ll_link, INT32_t * ll_val)