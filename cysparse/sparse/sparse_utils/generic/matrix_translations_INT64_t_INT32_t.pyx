#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.common_types.cysparse_types cimport *

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy

cdef csr_to_csc_kernel_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csr_ind, INT64_t * csr_col, INT32_t * csr_val,
                                      INT64_t * csc_ind, INT64_t * csc_row, INT32_t * csc_val):
    """
    Translate an CSR to an CSC matrix format.

    Args:
        nrow, ncol: Matrix dimension.
        nnz: Number of non zero elements.
        csr_ind, csr_col, csr_val: CSR matrix (IN argument).
        csc_ind, csc_row, csc_val: Computed CSC matrix (OUT argument).

    Note:
        This transformation **perserve** the row indices ordering, i.e. if the column indices of the CSR sparse matrix
        are ordered (ascending sorted), the row indices will be ordered (ascending sorted).
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



cdef csc_to_csr_kernel_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csc_ind, INT64_t * csc_row, INT32_t * csc_val,
                                      INT64_t * csr_ind, INT64_t * csr_col, INT32_t * csr_val):
    """
    Translate an CSR to an CSC matrix format.

    Args:
        nrow, ncol: Matrix dimension.
        nnz: Number of non zero elements.
        csc_ind, csc_row, csc_val: Computed CSC matrix (IN argument).
        csr_ind, csr_col, csr_val: CSR matrix (OUT argument).

    Note:
        This transformation **perserve** the column indices ordering, i.e. if the row indices of the CSC sparse matrix
        are ordered (ascending sorted), the column indices will be ordered (ascending sorted).
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


cdef csc_to_ll_kernel_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz,
                                      INT64_t * csc_ind, INT64_t * csc_row, INT32_t * csc_val,
                                      INT64_t * ll_root, INT64_t * ll_col, INT64_t * ll_link, INT32_t * ll_val):
    """
    Translate an CSR to an LL matrix format.

    Args:
        nrow, ncol: Matrix dimension.
        nnz: Number of non zero elements.
        csc_ind, csc_row, csc_val: Computed CSC matrix (IN argument).
        ll_root, ll_col, ll_link, ll_val: LL matrix (OUT argument).

    Note:
        This transformation **perserve** the column indices ordering, i.e. if the row indices of the CSC sparse matrix
        are ordered (ascending sorted), the column indices will be ordered (ascending sorted).
    """
    ############
    # first transform CSC to CSR
    ############
    # create CSR internal arrays: ind, col and val
    cdef INT64_t * ind = <INT64_t *> PyMem_Malloc((nrow + 1) * sizeof(INT64_t))
    if not ind:
        raise MemoryError()

    cdef INT64_t * col = <INT64_t *> PyMem_Malloc(nnz * sizeof(INT64_t))
    if not col:
        PyMem_Free(ind)
        raise MemoryError()

    cdef INT32_t * val = <INT32_t *> PyMem_Malloc(nnz * sizeof(INT32_t))
    if not val:
        PyMem_Free(ind)
        PyMem_Free(col)
        raise MemoryError()

    csc_to_csr_kernel_INT64_t_INT32_t(nrow, ncol, nnz, csc_ind, csc_row, csc_val, ind, col, val)

    ############
    # copy col[] and val[]
    ############
    memcpy(ll_col, col, nnz * sizeof(INT64_t))
    memcpy(ll_val, val, nnz * sizeof(INT32_t))

    ############
    # then compute root[] and link[]
    ############
    cdef INT64_t root_index = 0
    cdef INT64_t link_index = 0
    cdef INT64_t nbr_of_elements_in_row = 0
    cdef:
        INT64_t i, j

    for i from 0 <= i < nrow:
        nbr_of_elements_in_row = ind[i+1] - ind[i]

        if nbr_of_elements_in_row == 0:
            # row i is empty
            ll_root[i] = -1
        else:
            ll_root[i] = root_index

            for j from 0 <= j < nbr_of_elements_in_row - 1:
                link_index += 1
                ll_link[link_index - 1] = link_index

            ll_link[link_index] = -1
            link_index += 1

        root_index += nbr_of_elements_in_row