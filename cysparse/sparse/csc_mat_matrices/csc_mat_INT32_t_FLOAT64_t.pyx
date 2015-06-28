"""
Condensed Sparse Column (CSC) Format Matrices.


"""
from __future__ import print_function

from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat cimport unexposed_value

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_FLOAT64_t cimport ImmutableSparseMatrix_INT32_t_FLOAT64_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t cimport LLSparseMatrix_INT32_t_FLOAT64_t

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
cdef int CSC_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef int CSC_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format

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

include "csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_FLOAT64_t.pxi"
include "csc_mat_helpers/csc_mat_multiplication_INT32_t_FLOAT64_t.pxi"


cdef class CSCSparseMatrix_INT32_t_FLOAT64_t(ImmutableSparseMatrix_INT32_t_FLOAT64_t):
    """
    Compressed Sparse Column Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    def __cinit__(self,  **kwargs):

        self.__type = "CSCSparseMatrix"
        self.__type_name = "CSCSparseMatrix %s" % self.__index_and_type

    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.row)
        PyMem_Free(self.ind)

    def copy(self):
        """
        Return a (deep) copy of itself.

        Warning:
            Because we use memcpy and thus copy memory internally, we have to be careful to always update this method
            whenever the CSCSparseMatrix class changes.
        """
        # Warning: Because we use memcpy and thus copy memory internally, we have to be careful to always update this method
        # whenever the CSCSparseMatrix class changes...

        cdef CSCSparseMatrix_INT32_t_FLOAT64_t self_copy

        # we copy manually the C-arrays
        cdef:
            FLOAT64_t * val
            INT32_t * row
            INT32_t * ind
            INT32_t nnz

        nnz = self.nnz

        self_copy = CSCSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=self.__nrow, ncol=self.__ncol, store_zeros=self.__store_zeros, is_symmetric=self.__is_symmetric)

        val = <FLOAT64_t *> PyMem_Malloc(nnz * sizeof(FLOAT64_t))
        if not val:
            raise MemoryError()
        memcpy(val, self.val, nnz * sizeof(FLOAT64_t))
        self_copy.val = val

        row = <INT32_t *> PyMem_Malloc(nnz * sizeof(INT32_t))
        if not row:
            PyMem_Free(self_copy.val)
            raise MemoryError()
        memcpy(row, self.row, nnz * sizeof(INT32_t))
        self_copy.row = row

        ind = <INT32_t *> PyMem_Malloc((self.__ncol + 1) * sizeof(INT32_t))
        if not ind:
            PyMem_Free(self_copy.val)
            PyMem_Free(self_copy.row)
            raise MemoryError()
        memcpy(ind, self.ind, (self.__ncol + 1) * sizeof(INT32_t))
        self_copy.ind = ind

        self_copy.__nnz = nnz

        return self_copy

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

        # code is duplicated for optimization
        if self.__is_symmetric:
            # TODO: column indices are NOT necessarily sorted... what do we do about it?
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

    cdef FLOAT64_t safe_at(self, INT32_t i, INT32_t j) except? 2:

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

        cdef INT32_t i = key[0]
        cdef INT32_t j = key[1]

        return self.safe_at(i, j)

    def find(self):
        """
        Return 3 NumPy arrays with the non-zero matrix entries: i-rows, j-cols, vals.
        """
        pass
        cdef cnp.npy_intp dmat[1]
        dmat[0] = <cnp.npy_intp> self.__nnz

        # EXPLICIT TYPE TESTS

        cdef:
            cnp.ndarray[cnp.npy_int32, ndim=1] a_row = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)
            cnp.ndarray[cnp.npy_int32, ndim=1] a_col = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)
            cnp.ndarray[cnp.npy_float64, ndim=1] a_val = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_FLOAT64)

            INT32_t   *pi
            INT32_t   *pj
            FLOAT64_t    *pv
            INT32_t   j, k, elem

        pi = <INT32_t *> cnp.PyArray_DATA(a_row)
        pj = <INT32_t *> cnp.PyArray_DATA(a_col)
        pv = <FLOAT64_t *> cnp.PyArray_DATA(a_val)

        elem = 0
        for j from 0 <= j < self.__ncol:
            for k from self.ind[j] <= k < self.ind[j+1]:
                pi[ elem ] = self.row[j]
                pj[ elem ] = j
                pv[ elem ] = self.val[k]
                elem += 1

        return (a_row, a_col, a_val)


    ####################################################################################################################
    # Common operations
    ####################################################################################################################
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
            cnp.ndarray[cnp.npy_float64, ndim=1] diag = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_FLOAT64)
            FLOAT64_t    *pv
            INT32_t   i, j, k_

        pv = <FLOAT64_t *> cnp.PyArray_DATA(diag)

        # init NumPy array
        for i from 0 <= i < diag_size:

            pv[i] = 0.0


        if k >= 0:
            for j from 0 <= j < self.__ncol:
                for k_ from self.ind[j] <= k_ < self.ind[j+1]:
                    i = self.row[k_]
                    if i + k == j:
                        pv[i] = self.val[k_]

        else:  #  k < 0
            for j from 0 <= j < self.__ncol:
                for k_ from self.ind[j] <= k_ < self.ind[j+1]:
                    i = self.row[k_]
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
        return multiply_csc_mat_with_numpy_vector_INT32_t_FLOAT64_t(self, b)

    def matvec_transp(self, b):
        """
        Return :math:`A^t * b`.
        """
        return multiply_transposed_csc_mat_with_numpy_vector_INT32_t_FLOAT64_t(self, b)



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
        cdef INT32_t i, k, first = 1;

        cdef FLOAT64_t *mat
        cdef INT32_t j
        cdef FLOAT64_t val

        print('CSCSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz:
            return

        if self.nrow <= CSC_MAT_PPRINT_COL_THRESH and self.ncol <= CSC_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: Skip this creation
            mat = <FLOAT64_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(FLOAT64_t))

            if not mat:
                raise MemoryError()

            for j from 0 <= j < self.ncol:
                for i from 0 <= i < self.nrow:


                    mat[i* self.ncol + j] = 0.0


                # TODO: rewrite this completely
                for k from self.ind[j] <= k < self.ind[j+1]:
                    mat[(self.row[k]*self.ncol)+j] = self.val[k]
                    if self.__is_symmetric:
                        mat[(j*self.ncol)+ self.row[k]] = self.val[k]

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
        ind_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_INT32, <INT32_t *>self.ind)

        # row
        dim[0] = self.nnz
        row_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_INT32, <INT32_t *>self.row)

        # val
        dim[0] = self.nnz
        val_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_FLOAT64, <FLOAT64_t *>self.val)


        return ind_numpy_array, row_numpy_array, val_numpy_array

    ####################################################################################################################
    # DEBUG
    ####################################################################################################################
    def debug_print(self):
        cdef INT32_t i
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

    def set_row(self, INT32_t i, INT32_t val):
        self.row[i] = val




########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSCSparseMatrix_INT32_t_FLOAT64_t(INT32_t nrow, INT32_t ncol, INT32_t nnz, INT32_t * ind, INT32_t * row, FLOAT64_t * val, bint is_symmetric, bint store_zeros):
    """
    Construct a CSCSparseMatrix object.

    Args:
        nrow (INT32_t): Number of rows.
        ncol (INT32_t): Number of columns.
        nnz (INT32_t): Number of non-zeros.
        ind (INT32_t *): C-array with column indices pointers.
        row  (INT32_t *): C-array with row indices.
        val  (FLOAT64_t *): C-array with values.
    """
    csc_mat = CSCSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz, is_symmetric=is_symmetric, store_zeros=store_zeros)

    csc_mat.val = val
    csc_mat.ind = ind
    csc_mat.row = row

    return csc_mat
