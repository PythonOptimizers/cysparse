from sparse_lib.sparse.sparse_mat cimport MutableSparseMatrix

cimport numpy as cnp



cdef class LLSparseMatrix(MutableSparseMatrix):
    """
    Linked-List Format matrix.

    Note:
        Despite its name, this matrix doesn't use any linked list.
    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    cdef:
        int     free      # index to first element in free chain
        double *val       # pointer to array of values
        int    *col       # pointer to array of indices, see doc
        int    *link      # pointer to array of indices, see doc
        int    *root      # pointer to array of indices, see doc

    cdef _realloc(self, int nalloc_new)
    cdef _realloc_expand(self)

    cdef at(self, int i, int j)
    cdef safe_at(self, int i, int j)

cdef LLSparseMatrix multiply_two_ll_mat(LLSparseMatrix A, LLSparseMatrix B)

cdef multiply_ll_mat_with_numpy_ndarray(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=2] B)

cdef cnp.ndarray[cnp.double_t, ndim=1] multiply_ll_mat_with_numpy_vector(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=1, mode="c"] b)

cdef LLSparseMatrix transposed_ll_mat(LLSparseMatrix A)

cdef bint update_ll_mat_item_add(LLSparseMatrix A, int i, int j, double x)