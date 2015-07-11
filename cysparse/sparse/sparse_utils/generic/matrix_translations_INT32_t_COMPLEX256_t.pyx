from cysparse.types.cysparse_types cimport *


cdef csr_to_csc_kernel_INT32_t_COMPLEX256_t(INT32_t nrow, INT32_t ncol, INT32_t nnz,
                                      INT32_t * csr_ind, INT32_t * csr_col, COMPLEX256_t * csr_val,
                                      INT32_t * csc_ind, INT32_t * csc_row, COMPLEX256_t * csc_val):
    ############
    # compute csc_ind, i.e. the nnz of each column of the matrix
    ############
    cdef:
        INT32_t i, j, n

    # initialize to 0
    for j from 0 <= j <= ncol:
        csc_ind[j] = 0
    # count nnz per column
    for n from 0 <= n < nnz:
        csc_ind[csr_col[n]] += 1

    # cumsum the nnz per column to get csc_ind
    cdef:
        INT32_t cumsum = 0
        INT32_t temp = 0

    for j from 0<= j < ncol:
        temp  = csc_ind[j]
        csc_ind[j] = cumsum
        cumsum += temp

    csc_ind[ncol] = nnz

    # csc_ind is computed but will change in the next lines: we'll recompute it afterwoods

    ############
    # populate row and val
    ############
    cdef INT32_t jj, dest

    for i from 0 <= i < nrow:
        for jj from csr_ind[i] <= jj < csr_ind[i+1]:
            j  = csr_col[jj]
            dest = csc_ind[j]

            csc_row[dest] = i
            csc_val[dest] = csr_val[jj]

            csc_ind[j] += 1

    cdef INT32_t last = 0

    for j from 0 <= j <= ncol:
        temp   = csc_ind[j]
        csc_ind[j] = last
        last   = temp