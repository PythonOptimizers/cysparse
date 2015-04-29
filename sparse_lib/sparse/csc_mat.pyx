"""
Condensed Sparse Column (CSC) Format Matrices.


"""
from __future__ import print_function

from sparse_lib.cysparse_types cimport *

from sparse_lib.sparse.sparse_mat cimport ImmutableSparseMatrix, unexposed_value


from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef INT_t CSC_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef INT_t CSC_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format


cdef class CSCSparseMatrix(ImmutableSparseMatrix):
    """
    Compressed Sparse Column Format matrix.

    Note:
        This matrix can **not** be modified.

    """


    def __cinit__(self,  **kwargs):
        self.type_name = "CSCSparseMatrix"

    def __dealloc__(self):
        PyMem_Free(self.val)
        if self.is_complex:
            PyMem_Free(self.ival)
        PyMem_Free(self.row)
        PyMem_Free(self.ind)

    ####################################################################################################################
    # Set/Get items
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    def __setitem__(self, tuple key, value):
        raise SyntaxError("Assign individual elements is not allowed")

    #                                            *** GET ***
    cdef at(self, INT_t i, INT_t j):
        """
        Direct access to element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        cdef INT_t k

        if self.is_symmetric:
            raise NotImplemented("Access to csr_mat(i, j) not (yet) implemented")

        # TODO: column indices are NOT necessarily sorted... what do we do about it?
        for k from self.ind[j] <= k < self.ind[j+1]:
            if i == self.row[k]:
                return self.val[k]

        return 0.0

    cdef safe_at(self, INT_t i, INT_t j):
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

        cdef INT_t i = key[0]
        cdef INT_t j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def __repr__(self):
        s = "CSCSparseMatrix of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
        return s

    def print_to(self, OUT):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.
        """
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        cdef INT_t i, k, first = 1;

        cdef double *mat
        cdef INT_t j
        cdef double val

        print('CSCSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz:
            return

        if self.nrow <= CSC_MAT_PPRINT_COL_THRESH and self.ncol <= CSC_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: Skip this creation
            mat = <double *> PyMem_Malloc(self.nrow * self.ncol * sizeof(double))

            if not mat:
                raise MemoryError()

            for j from 0 <= j < self.ncol:
                for i from 0 <= i < self.nrow:
                    mat[j* self.nrow + i] = 0.0

                k = self.ind[j]
                while k < self.ind[j+1]:
                    mat[(j*self.nrow)+self.row[k]] = self.val[k]
                    k += 1

            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    val = mat[(j*self.nrow)+i]
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
        cdef INT_t i
        print("ind:")
        for i from 0 <= i < self.ncol + 1:
            print(self.ind[i], end=' ', sep=' ')
        print()

        print("row:")
        for i from 0 <= i < self.nnz:
            print(self.row[i], end=' ', sep=' ')
        print()

        print("val:")
        for i from 0 <= i < self.nnz:
            print(self.val[i], end=' == ', sep=' == ')
        print()

    def set_row(self, INT_t i, INT_t val):
        self.row[i] = val




########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSCSparseMatrix(INT_t nrow, INT_t ncol, INT_t nnz, INT_t * ind, INT_t * row, FLOAT_t * val):
    """
    Construct a CSCSparseMatrix object.

    Args:
        nrow (INT_t): Number of rows.
        ncol (INT_t): Number of columns.
        nnz (INT_t): Number of non-zeros.
        ind (INT_t *): C-array with column indices pointers.
        row  (INT_t *): C-array with row indices.
        val  (double *): C-array with values.
    """
    csc_mat = CSCSparseMatrix(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz)

    csc_mat.val = val
    csc_mat.ind = ind
    csc_mat.row = row

    return csc_mat

cdef MakeCSCComplexSparseMatrix(INT_t nrow, INT_t ncol, INT_t nnz, INT_t * ind, INT_t * row, FLOAT_t * val, FLOAT_t * ival):
    csc_mat = CSCSparseMatrix(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz)

    csc_mat.val = val
    csc_mat.ival = ival
    csc_mat.ind = ind
    csc_mat.row = row

    return csc_mat