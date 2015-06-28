"""
Condensed Sparse Row (CSR) Format Matrices.


"""

from __future__ import print_function

from cysparse.sparse.s_mat cimport unexposed_value
from cysparse.types.cysparse_numpy_types import *

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_INT32_t cimport ImmutableSparseMatrix_INT32_t_INT32_t, MutableSparseMatrix_INT32_t_INT32_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT32_t cimport LLSparseMatrix_INT32_t_INT32_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_INT32_t cimport CSCSparseMatrix_INT32_t_INT32_t

from cysparse.sparse.sparse_utils.generic.sort_indices_INT32_t cimport sort_array_INT32_t

########################################################################################################################
# Cython, NumPy import/cimport
########################################################################################################################
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc,free, calloc
from libc.string cimport memcpy
from cpython cimport PyObject, Py_INCREF


cimport numpy as cnp
import numpy as np

cnp.import_array()


# TODO: These constants will be removed soon...
cdef int CSR_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef int CSR_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format

cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)

    double creal(double complex z)
    double cimag(double complex z)

    long double creall(long double complex z)
    long double cimagl(long double complex z)

    double cabs(double complex z)
    float cabsf(float complex z)
    long double cabsl(long double complex z)

    double complex conj(double complex z)
    float complex  conjf (float complex z)
    long double complex conjl (long double complex z)

########################################################################################################################
# CySparse include
########################################################################################################################
# pxi files should come last (except for circular dependencies)

include "csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT32_t.pxi"
include "csr_mat_helpers/csr_mat_multiplication_INT32_t_INT32_t.pxi"


cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)



cdef class CSRSparseMatrix_INT32_t_INT32_t(ImmutableSparseMatrix_INT32_t_INT32_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    def __cinit__(self, **kwargs):

        self.__type = "CSRSparseMatrix"
        self.__type_name = "CSRSparseMatrix %s" % self.__index_and_type

    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.col)
        PyMem_Free(self.ind)

    def copy(self):
        """
        Return a (deep) copy of itself.

        Warning:
            Because we use memcpy and thus copy memory internally, we have to be careful to always update this method
            whenever the CSRSparseMatrix class changes.
        """
        # Warning: Because we use memcpy and thus copy memory internally, we have to be careful to always update this method
        # whenever the CSRSparseMatrix class changes...

        cdef CSRSparseMatrix_INT32_t_INT32_t self_copy

        # we copy manually the C-arrays
        cdef:
            INT32_t * val
            INT32_t * col
            INT32_t * ind
            INT32_t nnz

        nnz = self.nnz

        self_copy = CSRSparseMatrix_INT32_t_INT32_t(control_object=unexposed_value, nrow=self.__nrow, ncol=self.__ncol, store_zeros=self.__store_zeros, is_symmetric=self.__is_symmetric)

        val = <INT32_t *> PyMem_Malloc(nnz * sizeof(INT32_t))
        if not val:
            raise MemoryError()
        memcpy(val, self.val, nnz * sizeof(INT32_t))
        self_copy.val = val

        col = <INT32_t *> PyMem_Malloc(nnz * sizeof(INT32_t))
        if not col:
            PyMem_Free(self_copy.val)
            raise MemoryError()
        memcpy(col, self.col, nnz * sizeof(INT32_t))
        self_copy.col = col

        ind = <INT32_t *> PyMem_Malloc((self.__nrow + 1) * sizeof(INT32_t))
        if not ind:
            PyMem_Free(self_copy.val)
            PyMem_Free(self_copy.col)
            raise MemoryError()
        memcpy(ind, self.ind, (self.__nrow + 1) * sizeof(INT32_t))
        self_copy.ind = ind

        self_copy.__nnz = nnz

        return self_copy

    ####################################################################################################################
    # Column indices ordering
    ####################################################################################################################
    def are_column_indices_sorted(self):
        """
        Tell if column indices are sorted in augmenting order (ordered).


        """
        cdef INT32_t i
        cdef INT32_t col_index
        cdef INT32_t col_index_stop

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

        cdef INT32_t i = self.__first_row_not_ordered
        cdef INT32_t col_index
        cdef INT32_t col_index_start
        cdef INT32_t col_index_stop

        while i < self.nrow:
            col_index = self.ind[i]
            col_index_start = col_index
            col_index_stop = self.ind[i+1]

            while col_index < col_index_stop - 1:
                # detect if row is not ordered
                if self.col[col_index] > self.col[col_index + 1]:
                    # sort
                    # TODO: maybe use the column index for optimization?
                    sort_array_INT32_t(self.col, col_index_start, col_index_stop)
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
    cdef at(self, INT32_t i, INT32_t j):
        """
        Direct access to element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        cdef:
            INT32_t k
            # for symmetric case
            INT32_t real_i
            INT32_t real_j

        # TODO: TEST!!!
        # code duplicated for optimization
        if self.__is_symmetric:
            if i < j:
                real_i = j
                real_j = i
            else:
                real_i = i
                real_j = j

            if self. __col_indices_sorted:
                for k from self.ind[real_i] <= k < self.ind[real_i+1]:
                    if real_j == self.col[k]:
                        return self.val[k]
                    elif real_j > self.col[k]:
                        break

            else:
                for k from self.ind[real_i] <= k < self.ind[real_i+1]:
                    if real_j == self.col[k]:
                        return self.val[k]

        else:
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

    cdef INT32_t safe_at(self, INT32_t i, INT32_t j) except? 2:

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

        cdef INT32_t i = key[0]
        cdef INT32_t j = key[1]

        return self.safe_at(i, j)

    def find(self):
        """
        Return 3 NumPy arrays with the non-zero matrix entries: i-rows, j-cols, vals.
        """
        cdef cnp.npy_intp dmat[1]
        dmat[0] = <cnp.npy_intp> self.__nnz

        # EXPLICIT TYPE TESTS

        cdef:
            cnp.ndarray[cnp.npy_int32, ndim=1] a_row = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)
            cnp.ndarray[cnp.npy_int32, ndim=1] a_col = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)
            cnp.ndarray[cnp.npy_int32, ndim=1] a_val = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)

            INT32_t   *pi, *pj   # Intermediate pointers to matrix data
            INT32_t    *pv
            INT32_t   i, k, elem

        pi = <INT32_t *> cnp.PyArray_DATA(a_row)
        pj = <INT32_t *> cnp.PyArray_DATA(a_col)
        pv = <INT32_t *> cnp.PyArray_DATA(a_val)

        elem = 0
        for i from 0 <= i < self.__nrow:
            for k from self.ind[i] <= k < self.ind[i+1]:
                pi[ elem ] = i
                pj[ elem ] = self.col[k]
                pv[ elem ] = self.val[k]
                elem += 1

        return (a_row, a_col, a_val)

    def diag(self, k = 0):
        """
        Return the :math:`k^\textrm{th}` diagonal.

        """
        if not (-self.__nrow + 1 <= k <= self.__ncol -1):
            raise IndexError("Wrong diagonal number (%d <= k <= %d)" % (-self.__nrow + 1, self.__ncol -1))

        cdef INT32_t diag_size

        if k == 0:
            diag_size = min(self.__nrow, self.__ncol)
        elif k > 0:
            diag_size = min(self.__nrow, self.__ncol - k)
        else:
            diag_size = min(self.__nrow+k, self.__ncol)

        assert diag_size > 0, "Something is wrong with the diagonal size"

        # create NumPy array
        cdef cnp.npy_intp dmat[1]
        dmat[0] = <cnp.npy_intp> diag_size

        cdef:
            cnp.ndarray[cnp.npy_int32, ndim=1] diag = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)
            INT32_t    *pv
            INT32_t   i, k_

        pv = <INT32_t *> cnp.PyArray_DATA(diag)

        # init NumPy array
        for i from 0 <= i < diag_size:

            pv[i] = 0


        if k >= 0:
            for i from 0 <= i < self.__nrow:
                for k_ from self.ind[i] <= k_ < self.ind[i+1]:
                    if i + k == self.col[k_]:
                        pv[i] = self.val[k_]

        else:  #  k < 0
            for i from 0 <= i < self.__nrow:
                for k_ from self.ind[i] <= k_ < self.ind[i+1]:
                    j = self.col[k_]
                    if i + k == j:
                        pv[j] = self.val[k_]

        return diag


    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def matvec(self, b):
        """
        Return :math:`A * b`.
        """
        return multiply_csr_mat_with_numpy_vector_INT32_t_INT32_t(self, b)

    def matvec_transp(self, b):
        """
        Return :math:`A^t * b`.
        """
        return multiply_transposed_csr_mat_with_numpy_vector_INT32_t_INT32_t(self, b)



    def matdot(self, B):
        raise NotImplementedError("Multiplication with this kind of object not allowed")

    def matdot_transp(self, B):
        raise NotImplementedError("Multiplication with this kind of object not allowed")

    def __mul__(self, other):

        # test if implemented
        if isinstance(other, (MutableSparseMatrix_INT32_t_INT32_t, ImmutableSparseMatrix_INT32_t_INT32_t)):
            pass
        else:
            raise NotImplemented("Multiplication not (yet) allowed")

        # CASES
        if isinstance(other, CSCSparseMatrix_INT32_t_INT32_t):
            return multiply_csr_mat_by_csc_mat_INT32_t_INT32_t(self, other)
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
        cdef INT32_t i, k, first = 1;

        cdef INT32_t *mat
        cdef INT32_t j
        cdef INT32_t val

        print(self._matrix_description_before_printing(), file=OUT)
        #print('CSRSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz:
            return

        if self.nrow <= CSR_MAT_PPRINT_COL_THRESH and self.ncol <= CSR_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: put in a method of its own
            mat = <INT32_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(INT32_t))

            if not mat:
                raise MemoryError()

            # creation of temp matrix
            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:

                    mat[i* self.ncol + j] = 0


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
        cdef INT32_t i
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

    def set_col(self, INT32_t i, INT32_t val):
        self.col[i] = val

########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSRSparseMatrix_INT32_t_INT32_t(INT32_t nrow, INT32_t ncol, INT32_t nnz, INT32_t * ind, INT32_t * col, INT32_t * val, bint is_symmetric, bint store_zeros):
    """
    Construct a CSRSparseMatrix object.

    Args:
        nrow (INT32_t): Number of rows.
        ncol (INT32_t): Number of columns.
        nnz (INT32_t): Number of non-zeros.
        ind (INT32_t *): C-array with column indices pointers.
        col  (INT32_t *): C-array with column indices.
        val  (INT32_t *): C-array with values.
        is_symmetric (boolean): Is matrix symmetrix or not?
    """

    csr_mat = CSRSparseMatrix_INT32_t_INT32_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz, is_symmetric=is_symmetric, store_zeros=store_zeros)

    csr_mat.val = val
    csr_mat.ind = ind
    csr_mat.col = col

    return csr_mat

########################################################################################################################
# Multiplication functions
########################################################################################################################
# TODO: put in helpers...
cdef LLSparseMatrix_INT32_t_INT32_t multiply_csr_mat_by_csc_mat_INT32_t_INT32_t(CSRSparseMatrix_INT32_t_INT32_t A, CSCSparseMatrix_INT32_t_INT32_t B):

    if A.is_complex or B.is_complex:
        raise NotImplemented("This operation is not (yet) implemented for complex matrices")

    # TODO: take into account if matrix A or B has its column indices ordered or not...
    # test dimensions
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef INT32_t B_nrow = B.nrow
    cdef INT32_t B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT32_t C_nrow = A_nrow
    cdef INT32_t C_ncol = B_ncol

    cdef bint store_zeros = A.store_zeros and B.store_zeros
    # TODO: what strategy to implement?
    cdef INT32_t size_hint = A.nnz

    # TODO: maybe use MakeLLSparseMatrix and fix circular dependencies...
    C = LLSparseMatrix_INT32_t_INT32_t(control_object=unexposed_value, nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zeros=store_zeros)

    # CASES
    if not A.__is_symmetric and not B.__is_symmetric:
        pass
    else:
        raise NotImplemented("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    # TODO: what do we do? Column indices are NOT necessarily sorted...
    cdef:
        INT32_t i, j, k
        INT32_t sum

    # don't keep zeros, no matter what
    cdef bint old_store_zeros = store_zeros
    C.store_zeros = 0

    for i from 0 <= i < C_nrow:
        for j from 0 <= j < C_ncol:

            sum = 0


            for k from 0 <= k < A_ncol:
                sum += (A[i, k] * B[k, j])

            C.put(i, j, sum)

    C.store_zeros = old_store_zeros

    return C