"""
Diverse utilities to translate one matrix format into another.

"""
from cysparse.types.cysparse_types cimport *

cdef csr_to_csc_kernel_INT32_t_INT32_t(INT32_t nrow, INT32_t ncol, INT32_t nnz,
                                      INT32_t * csr_ind, INT32_t * csr_col, INT32_t * csr_val,
                                      INT32_t * csc_ind, INT32_t * csc_row, INT32_t * csc_val)