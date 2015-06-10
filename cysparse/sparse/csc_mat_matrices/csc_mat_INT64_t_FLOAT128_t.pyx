"""
Condensed Sparse Column (CSC) Format Matrices.


"""
from __future__ import print_function

from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat cimport unexposed_value

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_FLOAT128_t cimport ImmutableSparseMatrix_INT64_t_FLOAT128_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT128_t cimport LLSparseMatrix_INT64_t_FLOAT128_t

########################################################################################################################
# Cython, NumPy import/cimport
########################################################################################################################
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from python_ref cimport Py_INCREF, Py_DECREF

cimport numpy as cnp
import numpy as np

cnp.import_array()

# TODO: These constants will be removed soon...
cdef int CSC_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef int CSC_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format


########################################################################################################################
# CySparse include
########################################################################################################################
# pxi files should come last (except for circular dependencies)

include "csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT64_t_FLOAT128_t.pxi"
include "csc_mat_helpers/csc_mat_multiplication_INT64_t_FLOAT128_t.pxi"


cdef class CSCSparseMatrix_INT64_t_FLOAT128_t(ImmutableSparseMatrix_INT64_t_FLOAT128_t):
    """
    Compressed Sparse Column Format matrix.

    Note:
        This matrix can **not** be modified.

    """


    def __cinit__(self,  **kwargs):
        self.type_name = "CSCSparseMatrix"

    def __dealloc__(self):
        PyMem_Free(self.val)
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
    cdef at(self, INT64_t i, INT64_t j):
        """
        Direct access to element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        cdef:
            INT64_t k
            # for symmetric case
            INT64_t real_i
            INT64_t real_j

        if self.is_symmetric:
            # TODO: column indices are NOT necessarily sorted... what do we do about it?
            #raise NotImplementedError("Access to csc_mat(i, j) not (yet) implemented")
            if i < j:
                real_i = j
                real_j = i
            else:
                real_i = i
                real_j = j

            for k from self.ind[real_j] <= k < self.ind[real_j+1]:
                if real_i == self.row[k]:
                    return self.val[k]

        else:  # not symmetric
            # TODO: column indices are NOT necessarily sorted... what do we do about it?
            for k from self.ind[j] <= k < self.ind[j+1]:
                if i == self.row[k]:
                    return self.val[k]

        return 0.0

    # EXPLICIT TYPE TESTS

    cdef FLOAT128_t safe_at(self, INT64_t i, INT64_t j) except? 2:

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

        cdef INT64_t i = key[0]
        cdef INT64_t j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    # Common operations
    ####################################################################################################################
    def diag(self):
        """
        Return diagonal in a :program:`NumPy` array.
        """
        # TODO: write version when indices are sorted
        # TODO: refactor: act only on C-array? This would give the possibility to export NumPy array or C-array pointer

        cdef INT64_t diag_size = min(self.nrow, self.ncol)
        cdef cnp.ndarray[cnp.npy_float128, ndim=1, mode='c'] diagonal = np.zeros(diag_size, dtype=np.float128)

        # direct access to NumPy array
        cdef FLOAT128_t * diagonal_data = <FLOAT128_t *> cnp.PyArray_DATA(diagonal)

        cdef INT64_t j, k

        for j from 0 <= j < self.ncol:

            k = self.ind[j]

            while k < self.ind[j+1]:
                if self.row[k] == j:
                    # we have found the diagonal element
                    diagonal_data[j] = self.val[k]

                k += 1

        return diagonal

    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def matvec(self, B):
        """
        Return :math:`A * b`.
        """
        return multiply_csc_mat_with_numpy_vector_INT64_t_FLOAT128_t(self, B)

    def matvec_transp(self, B):
        """
        Return :math:`A^t * b`.
        """
        return multiply_transposed_csc_mat_with_numpy_vector_INT64_t_FLOAT128_t(self, B)

    def matdot(self, B):
        raise NotImplementedError("Multiplication with this kind of object not allowed")

    def matdot_transp(self, B):
        raise NotImplementedError("Multiplication with this kind of object not allowed")

    def __mul__(self, B):
        """
        Return :math:`A * B`.

        """
        if cnp.PyArray_Check(B) and B.ndim == 1:
            return self.matvec(B)

        return self.matdot(B)

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
        cdef INT64_t i, k, first = 1;

        cdef FLOAT128_t *mat
        cdef INT64_t j
        cdef FLOAT128_t val

        print('CSCSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz:
            return

        if self.nrow <= CSC_MAT_PPRINT_COL_THRESH and self.ncol <= CSC_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: Skip this creation
            mat = <FLOAT128_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(FLOAT128_t))

            if not mat:
                raise MemoryError()

            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:

                    mat[i* self.nrow + j] = 0.0


                    # BUG: this is non sense as it is computed for every different i
                    # TODO: rewrite this completely
                    k = self.ind[j]
                    while k < self.ind[j+1]:
                        mat[(self.row[k]*self.nrow)+j] = self.val[k]
                        if self.is_symmetric:
                            mat[(j*self.nrow)+ self.row[k]] = self.val[k]
                        k += 1

            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    val = mat[(i*self.nrow)+j]
                    #print('%9.*f ' % (6, val), file=OUT, end='')
                    print('{0:9.6f} '.format(val), end='')
                print()

            PyMem_Free(mat)

        else:
            print('Matrix too big to print out', file=OUT)

    ####################################################################################################################
    # Access to internals as resquested by Sylvain
    #
    # This is temporary and shouldn't be released!!!!
    #
    ####################################################################################################################
    def get_c_pointers(self):
        """
        Return C pointers to internal arrays.

        Returns:
            Triple `(ind, row, val)`.

        Warning:
            The returned values can only be used by C-extensions.
        """
        cdef:
            PyObject * ind_obj = <PyObject *> self.ind
            PyObject * row_obj = <PyObject *> self.row
            PyObject * val_obj = <PyObject *> self.val

        return <object>ind_obj, <object>row_obj, <object>val_obj

    def get_numpy_arrays(self):
        """
        Return :program:`NumPy` arrays equivalent to internal C-arrays.

        Note:
            No copy is made, i.e. the :program:`NumPy` arrays have direct access to the internal C-arrays. Change the
            former and you change the latter (which shouldn't happen unless you **really** know what you are doing).
        """
        cdef:
            cnp.npy_intp dim[1]

        # ind
        dim[0] = self.ncol + 1
        ind_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_INT64, <INT64_t *>self.ind)

        # row
        dim[0] = self.nnz
        row_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_INT64, <INT64_t *>self.row)

        # val
        dim[0] = self.nnz
        val_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_FLOAT128, <FLOAT128_t *>self.val)


        return ind_numpy_array, row_numpy_array, val_numpy_array

    ####################################################################################################################
    # DEBUG
    ####################################################################################################################
    def debug_print(self):
        cdef INT64_t i
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

    def set_row(self, INT64_t i, INT64_t val):
        self.row[i] = val




########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSCSparseMatrix_INT64_t_FLOAT128_t(INT64_t nrow, INT64_t ncol, INT64_t nnz, INT64_t * ind, INT64_t * row, FLOAT128_t * val, bint is_symmetric):
    """
    Construct a CSCSparseMatrix object.

    Args:
        nrow (INT64_t): Number of rows.
        ncol (INT64_t): Number of columns.
        nnz (INT64_t): Number of non-zeros.
        ind (INT64_t *): C-array with column indices pointers.
        row  (INT64_t *): C-array with row indices.
        val  (FLOAT128_t *): C-array with values.
    """
    csc_mat = CSCSparseMatrix_INT64_t_FLOAT128_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz, is_symmetric=is_symmetric)

    csc_mat.val = val
    csc_mat.ind = ind
    csc_mat.row = row

    return csc_mat
