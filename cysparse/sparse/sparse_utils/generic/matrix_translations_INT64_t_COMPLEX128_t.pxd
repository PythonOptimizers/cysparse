"""
Diverse utilities to translate one matrix format into another.

"""
from cysparse.types.cysparse_types cimport *

cdef csr_to_csc_kernel_INT64_t_COMPLEX128_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csr_ind, INT64_t * csr_col, COMPLEX128_t * csr_val,
                                      INT64_t * csc_ind, INT64_t * csc_row, COMPLEX128_t * csc_val)

cdef csc_to_csr_kernel_INT64_t_COMPLEX128_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csc_ind, INT64_t * csc_row, COMPLEX128_t * csc_val,
                                      INT64_t * csr_ind, INT64_t * csr_col, COMPLEX128_t * csr_val)