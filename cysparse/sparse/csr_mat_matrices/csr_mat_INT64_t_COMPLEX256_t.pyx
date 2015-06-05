"""
Condensed Sparse Row (CSR) Format Matrices.


"""

from __future__ import print_function

from cysparse.sparse.s_mat cimport unexposed_value

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX256_t cimport ImmutableSparseMatrix_INT64_t_COMPLEX256_t, MutableSparseMatrix_INT64_t_COMPLEX256_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX256_t cimport LLSparseMatrix_INT64_t_COMPLEX256_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX256_t cimport CSCSparseMatrix_INT64_t_COMPLEX256_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject

cdef int CSR_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef int CSR_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format

cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cdef _sort(INT64_t * a, INT64_t start, INT64_t end):
    """
    Sort array a between start and end - 1 (i.e. end **not** included).


    """
    # TODO: put this is a new file and test
    cdef INT64_t i, j, value;

    i = start

    while i < end:

        value = a[i]
        j = i - 1
        while j >= start and a[j] > value:
            # shift
            a[j+1] = a[j]

            j -= 1

        # place key at right place
        a[j+1] = value

        i += 1

cdef class CSRSparseMatrix_INT64_t_COMPLEX256_t(ImmutableSparseMatrix_INT64_t_COMPLEX256_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    def __cinit__(self, **kwargs):

        self.type_name = "CSRSparseMatrix"

    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.col)
        PyMem_Free(self.ind)

    ####################################################################################################################
    # Column indices ordering
    ####################################################################################################################
    def are_column_indices_sorted(self):
        """
        Tell if column indices are sorted in augmenting order (ordered).


        """
        cdef INT64_t i
        cdef INT64_t col_index
        cdef INT64_t col_index_stop

        if self.__col_indices_sorted_test_done:
            return self.__col_indices_sorted
        else:
            # do the test
            self.__col_indices_sorted_test_done = True
            # test each row
            for i from 0 <= i < self.nrow:
                col_index = self.ind[i]
                col_index_stop = self.ind[i+1] - 1

                self.__first_row_not_ordered = i

                while col_index < col_index_stop:
                    if self.col[col_index] > self.col[col_index + 1]:
                        self.__col_indices_sorted = False
                        return self.__col_indices_sorted
                    col_index += 1

        # column indices are ordered
        self.__first_row_not_ordered = self.nrow
        self.__col_indices_sorted = True
        return self.__col_indices_sorted

    cdef _order_column_indices(self):
        """
        Order column indices by ascending order.

        We use a simple insert sort. The idea is that the column indices aren't that much not ordered.
        """
        #  must be called to find first row not ordered
        if self.are_column_indices_sorted():
            return

        cdef INT64_t i = self.__first_row_not_ordered
        cdef INT64_t col_index
        cdef INT64_t col_index_start
        cdef INT64_t col_index_stop

        while i < self.nrow:
            col_index = self.ind[i]
            col_index_start = col_index
            col_index_stop = self.ind[i+1]

            while col_index < col_index_stop - 1:
                # detect if row is not ordered
                if self.col[col_index] > self.col[col_index + 1]:
                    # sort
                    # TODO: maybe use the column index for optimization?
                    _sort(self.col, col_index_start, col_index_stop)
                    break
                else:
                    col_index += 1

            i += 1

    def order_column_indices(self):
        """
        Forces column indices to be ordered.
        """
        return self._order_column_indices()

    cdef _set_column_indices_ordered_is_true(self):
        """
        If you construct a CSR matrix and you know that its column indices **are** ordered, confirm it by calling this method.

        Warning:
            Be sure to know what you are doing because there is no control and we assume that the column indices are indeed sorted for
            almost all operations.
        """
        self.__col_indices_sorted_test_done = True
        self.__col_indices_sorted = True

    ####################################################################################################################
    # Set/Get items
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    def __setitem__(self, tuple key, value):
        raise SyntaxError("Assign individual elements is not allowed")

    #                                            *** GET ***
    cdef at(self, INT64_t i, INT64_t j):
        """
        Direct access to element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        cdef INT64_t k

        if self.is_symmetric:
            raise NotImplemented("Access to csr_mat(i, j) not (yet) implemented for symmetric matrices")

        # TODO: TEST!!!
        if self. __col_indices_sorted:
            for k from self.ind[i] <= k < self.ind[i+1]:
                if j == self.col[k]:
                    return self.val[k]
                elif j > self.col[k]:
                    break

        else:
            for k from self.ind[i] <= k < self.ind[i+1]:
                if j == self.col[k]:
                    return self.val[k]

        return 0.0

    # EXPLICIT TYPE TESTS

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX256_t safe_at(self, INT64_t i, INT64_t j) except *:

        """
        Return element ``(i, j)`` but with check for out of bounds indices.

        Raises:
            IndexError: when index out of bound.

        """
        if not 0 <= i < self.nrow or not 0 <= j < self.ncol:
            raise IndexError("Index out of bounds")

        return self.at(i, j)

    def __getitem__(self, tuple key):
        """
        Return ``csr_mat[i, j]``.

        Args:
          key = (i,j): Must be a couple of integers.

        Raises:
            IndexError: when index out of bound.

        Returns:
            ``csr_mat[i, j]``.
        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integer indices are allowed")

        cdef INT64_t i = key[0]
        cdef INT64_t j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def __mul__(self, other):

        # test if implemented
        if isinstance(other, (MutableSparseMatrix_INT64_t_COMPLEX256_t, ImmutableSparseMatrix_INT64_t_COMPLEX256_t)):
            pass
        else:
            raise NotImplemented("Multiplication not (yet) allowed")

        # CASES
        if isinstance(other, CSCSparseMatrix_INT64_t_COMPLEX256_t):
            return multiply_csr_mat_by_csc_mat_INT64_t_COMPLEX256_t(self, other)
        else:
            raise NotImplemented("Multiplication not (yet) allowed")

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def __repr__(self):
        s = "CSRSparseMatrix of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
        return s

    def print_to(self, OUT, width=9, print_big_matrices=False, transposed=False):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.
        """
        # EXPLICIT TYPE TESTS
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        cdef INT64_t i, k, first = 1;

        cdef COMPLEX256_t *mat
        cdef INT64_t j
        cdef COMPLEX256_t val

        print(self._matrix_description_before_printing(), file=OUT)
        #print('CSRSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz:
            return

        if self.nrow <= CSR_MAT_PPRINT_COL_THRESH and self.ncol <= CSR_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: put in a method of its own
            mat = <COMPLEX256_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(COMPLEX256_t))

            if not mat:
                raise MemoryError()

            # creation of temp matrix
            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:

                    mat[i* self.ncol + j] = 0.0 + 0.0j


                k = self.ind[i]
                while k < self.ind[i+1]:
                    mat[(i*self.ncol)+self.col[k]] = self.val[k]
                    k += 1

            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    val = mat[(i*self.ncol)+j]

                    print('{:{width}.6f} '.format(val, width=width), end='', file=OUT)

                print(file=OUT)

            PyMem_Free(mat)

        else:
            print('Matrix too big to print out', file=OUT)

    ####################################################################################################################
    # DEBUG
    ####################################################################################################################
    def debug_print(self):
        cdef INT64_t i
        print("ind:")
        for i from 0 <= i < self.nrow + 1:
            print(self.ind[i], end=' ', sep=' ')
        print()

        print("col:")
        for i from 0 <= i < self.nnz:
            print(self.col[i], end=' ', sep=' ')
        print()

        print("val:")
        for i from 0 <= i < self.nnz:
            print(self.val[i], end=' == ', sep=' == ')
        print()

        if self.is_complex:
            for i from 0 <= i < self.nnz:
                print(self.ival[i], end=' == ', sep=' == ')
            print()

    def set_col(self, INT64_t i, INT64_t val):
        self.col[i] = val

########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSRSparseMatrix_INT64_t_COMPLEX256_t(INT64_t nrow, INT64_t ncol, INT64_t nnz, INT64_t * ind, INT64_t * col, COMPLEX256_t * val):
    """
    Construct a CSRSparseMatrix object.

    Args:
        nrow (INT64_t): Number of rows.
        ncol (INT64_t): Number of columns.
        nnz (INT64_t): Number of non-zeros.
        ind (INT64_t *): C-array with column indices pointers.
        col  (INT64_t *): C-array with column indices.
        val  (COMPLEX256_t *): C-array with values.
    """

    csr_mat = CSRSparseMatrix_INT64_t_COMPLEX256_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz)

    csr_mat.val = val
    csr_mat.ind = ind
    csr_mat.col = col

    return csr_mat

########################################################################################################################
# Multiplication functions
########################################################################################################################
cdef LLSparseMatrix_INT64_t_COMPLEX256_t multiply_csr_mat_by_csc_mat_INT64_t_COMPLEX256_t(CSRSparseMatrix_INT64_t_COMPLEX256_t A, CSCSparseMatrix_INT64_t_COMPLEX256_t B):

    if A.is_complex or B.is_complex:
        raise NotImplemented("This operation is not (yet) implemented for complex matrices")

    # TODO: take into account if matrix A or B has its column indices ordered or not...
    # test dimensions
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef INT64_t B_nrow = B.nrow
    cdef INT64_t B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT64_t C_nrow = A_nrow
    cdef INT64_t C_ncol = B_ncol

    cdef bint store_zeros = A.store_zeros and B.store_zeros
    # TODO: what strategy to implement?
    cdef INT64_t size_hint = A.nnz

    # TODO: maybe use MakeLLSparseMatrix and fix circular dependencies...
    C = LLSparseMatrix_INT64_t_COMPLEX256_t(control_object=unexposed_value, nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zeros=store_zeros)

    # CASES
    if not A.is_symmetric and not B.is_symmetric:
        pass
    else:
        raise NotImplemented("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    # TODO: what do we do? Column indices are NOT necessarily sorted...
    cdef:
        INT64_t i, j, k
        COMPLEX256_t sum

    # don't keep zeros, no matter what
    cdef bint old_store_zeros = store_zeros
    C.store_zeros = 0

    for i from 0 <= i < C_nrow:
        for j from 0 <= j < C_ncol:

            sum = 0.0 + 0.0j


            for k from 0 <= k < A_ncol:
                sum += (A[i, k] * B[k, j])

            C.put(i, j, sum)

    C.store_zeros = old_store_zeros

    return C