"""
Condensed Sparse Row (CSR) Format Matrices.


"""

from __future__ import print_function

from sparse_lib.sparse.sparse_mat cimport ImmutableSparseMatrix, MutableSparseMatrix
from sparse_lib.sparse.ll_mat cimport LLSparseMatrix
from sparse_lib.sparse.csc_mat cimport CSCSparseMatrix

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef int CSR_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef int CSR_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format


cdef _sort(int * a, int start, int end):
    """
    Sort array a between start and end - 1 (i.e. end **not** included).


    """
    # TODO: put this is a new file and test
    cdef int i, j, value;

    print ("start= %d, end = %d" % (start, end))
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


cdef class CSRSparseMatrix(ImmutableSparseMatrix):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    def __cinit__(self, int nrow, int ncol, int nnz):
        self.__status_ok = False

    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.col)
        PyMem_Free(self.ind)

    ####################################################################################################################
    # Control
    ####################################################################################################################
    def is_well_constructed(self):
        """
        Tell if object is well constructed, i.e. if it was instantiated by a factory method.

        This method **doesn't** test if the matrix is coherent, only if its memory was properly initialized with a
        factory method.

        Returns:
            ``(status, error_msg)``:

        """
        error_msg = None
        well_constructed_ok = True

        if not self.__status_ok:
            error_msg = "CSRSparseMatrix must be instantiated by a factory method!"
            well_constructed_ok = self.__status_ok

        return well_constructed_ok, error_msg

    def raise_error_if_not_well_constructed(self):
        """
        Raises an error if method :meth:`is_well_constructed()` doesn't return ``True``.

        See:
            :meth:`is_well_constructed`.

        Raises:
            NotImplementedError: If the private attribute ``__status_ok`` is not ``True``.
        """
        status_ok, error_msg = self.is_well_constructed()

        if not status_ok:
            raise NotImplementedError(error_msg)

    ####################################################################################################################
    # Column indices ordering
    ####################################################################################################################
    def are_column_indices_sorted(self):
        """
        Tell if column indices are sorted in augmenting order.


        """
        cdef int i
        cdef int col_index
        cdef int col_index_stop

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

        cdef int i = self.__first_row_not_ordered
        cdef int col_index
        cdef int col_index_start
        cdef int col_index_stop

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
        return self._order_column_indices()


    ####################################################################################################################
    # Set/Get items
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    def __setitem__(self, tuple key, value):
        raise SyntaxError("Assign individual elements is not allowed")

    #                                            *** GET ***
    cdef at(self, int i, int j):
        """
        Direct access to element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        cdef int k

        if self.is_symmetric:
            raise NotImplemented("Access to csr_mat(i, j) not (yet) implemented")

        # TODO: column indices are NOT necessarily sorted... what do we do about it?
        for k from self.ind[i] <= k < self.ind[i+1]:
            if j == self.col[k]:
                return self.val[k]

        return 0.0

    cdef safe_at(self, int i, int j):
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

        cdef int i = key[0]
        cdef int j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def __mul__(self, other):

        # test if implemented
        if isinstance(other, (MutableSparseMatrix, ImmutableSparseMatrix)):
            pass
        else:
            raise NotImplemented("Multiplication not (yet) allowed")

        # CASES
        if isinstance(other, CSCSparseMatrix):
            return multiply_csr_mat_by_csc_mat(self, other)

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def __repr__(self):
        s = "CSRSparseMatrix of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
        return s

    def print_to(self, OUT):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.
        """
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        cdef int i, k, first = 1;

        cdef double *mat
        cdef int j
        cdef double val

        print('CSRSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz or not self.__status_ok:
            return

        if self.nrow <= CSR_MAT_PPRINT_COL_THRESH and self.ncol <= CSR_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: put in a method of its own
            mat = <double *> PyMem_Malloc(self.nrow * self.ncol * sizeof(double))

            if not mat:
                raise MemoryError()

            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    mat[i* self.ncol + j] = 0.0

                k = self.ind[i]
                while k < self.ind[i+1]:
                    mat[(i*self.ncol)+self.col[k]] = self.val[k]
                    k += 1

            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    val = mat[(i*self.ncol)+j]
                    #print('%9.*f ' % (6, val), file=OUT, end='')
                    print('{0:9.6f} '.format(val), end='')
                print()

            PyMem_Free(mat)

        else:
            print('Matrix too big to print out', file=OUT)

    ####################################################################################################################
    # DEBUG
    ####################################################################################################################
    def debug_print(self):
        cdef int i
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

    def set_col(self, int i, int val):
        self.col[i] = val

########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSRSparseMatrix(int nrow, int ncol, int nnz, int * ind, int * col, double * val):
    """
    Construct a CSRSparseMatrix object.

    Args:
        nrow (int): Number of rows.
        ncol (int): Number of columns.
        nnz (int): Number of non-zeros.
        ind (int *): C-array with column indices pointers.
        col  (int *): C-array with column indices.
        val  (double *): C-array with values.
    """


    csr_mat = CSRSparseMatrix(nrow=nrow, ncol=ncol, nnz=nnz)

    csr_mat.val = val
    csr_mat.ind = ind
    csr_mat.col = col

    csr_mat.__status_ok = True

    return csr_mat

########################################################################################################################
# Multiplication functions
########################################################################################################################
cdef LLSparseMatrix multiply_csr_mat_by_csc_mat(CSRSparseMatrix A, CSCSparseMatrix B):
    # test dimensions
    cdef int A_nrow = A.nrow
    cdef int A_ncol = A.ncol

    cdef int B_nrow = B.nrow
    cdef int B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef int C_nrow = A_nrow
    cdef int C_ncol = B_ncol

    cdef bint store_zeros = A.store_zeros and B.store_zeros
    # TODO: what strategy to implement?
    cdef int size_hint = A.nnz

    C = LLSparseMatrix(nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zeros=store_zeros)

    # CASES
    if not A.is_symmetric and not B.is_symmetric:
        pass
    else:
        raise NotImplemented("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    # TODO: what do we do? Column indices are NOT necessarily sorted...
    cdef:
        int i, j, k
        double sum

    # don't keep zeros, no matter what
    cdef bint old_store_zeros = store_zeros
    C.store_zeros = 0

    for i from 0 <= i < C_nrow:
        for j from 0 <= j < C_ncol:
            sum = 0.0

            for k from 0 <= k < A_ncol:
                sum += (A[i, k] * B[k, j])

            C.put(i, j, sum)

    C.store_zeros = old_store_zeros

    return C