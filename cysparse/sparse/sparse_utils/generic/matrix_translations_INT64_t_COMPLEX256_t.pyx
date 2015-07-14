from cysparse.types.cysparse_types cimport *


cdef csr_to_csc_kernel_INT64_t_COMPLEX256_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csr_ind, INT64_t * csr_col, COMPLEX256_t * csr_val,
                                      INT64_t * csc_ind, INT64_t * csc_row, COMPLEX256_t * csc_val):
    """
    Translate an CSR to an CSC matrix format.

    Args:
        nrow, ncol: Matrix dimension.
        nnz: Number of non zero elements.
        csr_ind, csr_col, csr_val: CSR matrix (IN argument).
        csc_ind, csc_row, csc_val: Computed CSC matrix (OUT argument).
    """
    ############
    # compute csc_ind, i.e. the nnz of each column of the matrix
    ############
    cdef:
        INT64_t i, j, n

    # initialize to 0
    for j from 0 <= j <= ncol:
        csc_ind[j] = 0
    # count nnz per column
    for n from 0 <= n < nnz:
        csc_ind[csr_col[n]] += 1

    # cumsum the nnz per column to get csc_ind
    cdef:
        INT64_t cumsum = 0
        INT64_t temp = 0

    for j from 0<= j < ncol:
        temp  = csc_ind[j]
        csc_ind[j] = cumsum
        cumsum += temp

    csc_ind[ncol] = nnz

    # csc_ind is computed but will change in the next lines: we'll recompute it afterwoods
    # csc_ind will be shifted to the left, for instance [0, 1, 2, 3] becomes [1, 2, 3, 3]

    ############
    # populate row and val
    ############
    cdef INT64_t jj, dest

    for i from 0 <= i < nrow:
        for jj from csr_ind[i] <= jj < csr_ind[i+1]:
            j  = csr_col[jj]
            dest = csc_ind[j]

            csc_row[dest] = i
            csc_val[dest] = csr_val[jj]

            csc_ind[j] += 1

    # reshift csc_ind to the right, starting with csc_ind[0] = 0
    # from [1, 2, 3, 3] get back [0, 1, 2 , 3]

    cdef INT64_t last = 0

    for j from 0 <= j <= ncol:
        temp   = csc_ind[j]
        csc_ind[j] = last
        last   = temp



cdef csc_to_csr_kernel_INT64_t_COMPLEX256_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csc_ind, INT64_t * csc_row, COMPLEX256_t * csc_val,
                                      INT64_t * csr_ind, INT64_t * csr_col, COMPLEX256_t * csr_val):
    """
    Translate an CSR to an CSC matrix format.

    Args:
        nrow, ncol: Matrix dimension.
        nnz: Number of non zero elements.
        csc_ind, csc_row, csc_val: Computed CSC matrix (IN argument).
        csr_ind, csr_col, csr_val: CSR matrix (OUT argument).
    """
    ############
    # compute csr_ind, i.e. the nnz of each row of the matrix
    ############
    cdef:
        INT64_t i, j, n

    # initialize to 0
    for i from 0 <= i <= nrow:
        csr_ind[i] = 0
    # count nnz per row
    for n from 0 <= n < nnz:
        csr_ind[csc_row[n]] += 1

    # cumsum the nnz per row to get csr_ind
    cdef:
        INT64_t cumsum = 0
        INT64_t temp = 0

    for i from 0<= i < nrow:
        temp  = csr_ind[i]
        csr_ind[i] = cumsum
        cumsum += temp

    csr_ind[nrow] = nnz

    # csr_ind is computed but will change in the next lines: we'll recompute it afterwoods
    # csr_ind will be shifted to the left, for instance [0, 1, 2, 3] becomes [1, 2, 3, 3]

    ############
    # populate col and val
    ############
    cdef INT64_t ii, dest

    for j from 0 <= j < ncol:
        for ii from csc_ind[j] <= ii < csc_ind[j+1]:
            i  = csc_row[ii]
            dest = csr_ind[i]

            csr_col[dest] = j
            csr_val[dest] = csc_val[ii]

            csr_ind[i] += 1

    # reshift csr_ind to the right, starting with csr_ind[0] = 0
    # from [1, 2, 3, 3] get back [0, 1, 2 , 3]

    cdef INT64_t last = 0

    for i from 0 <= i <= nrow:
        temp   = csr_ind[i]
        csr_ind[i] = last
        last   = temp
