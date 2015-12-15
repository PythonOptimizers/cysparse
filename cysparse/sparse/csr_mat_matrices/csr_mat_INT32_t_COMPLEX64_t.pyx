"""
Condensed Sparse Row (CSR) Format Matrices.


"""

from __future__ import print_function

from cysparse.sparse.s_mat cimport unexposed_value
from cysparse.cysparse_types.cysparse_numpy_types import *

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_COMPLEX64_t cimport ImmutableSparseMatrix_INT32_t_COMPLEX64_t, MutableSparseMatrix_INT32_t_COMPLEX64_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX64_t cimport LLSparseMatrix_INT32_t_COMPLEX64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_COMPLEX64_t cimport CSCSparseMatrix_INT32_t_COMPLEX64_t, MakeCSCSparseMatrix_INT32_t_COMPLEX64_t

from cysparse.sparse.sparse_utils.generic.sort_indices_INT32_t cimport sort_array_INT32_t
from cysparse.sparse.sparse_utils.generic.print_COMPLEX64_t cimport element_to_string_COMPLEX64_t, conjugated_element_to_string_COMPLEX64_t, empty_to_string_COMPLEX64_t
from cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_COMPLEX64_t cimport csr_to_csc_kernel_INT32_t_COMPLEX64_t, csc_to_csr_kernel_INT32_t_COMPLEX64_t

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

include "csr_mat_kernel/csr_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX64_t.pxi"
include "csr_mat_helpers/csr_mat_multiplication_INT32_t_COMPLEX64_t.pxi"


cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)



cdef class CSRSparseMatrix_INT32_t_COMPLEX64_t(ImmutableSparseMatrix_INT32_t_COMPLEX64_t):
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

        self.__col_indices_sorted_test_done = False
        self.__col_indices_sorted = False
        self.__first_row_not_ordered = -1

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

        cdef CSRSparseMatrix_INT32_t_COMPLEX64_t self_copy

        # we copy manually the C-arrays
        cdef:
            COMPLEX64_t * val
            INT32_t * col
            INT32_t * ind
            INT32_t nnz

        nnz = self.nnz

        self_copy = CSRSparseMatrix_INT32_t_COMPLEX64_t(control_object=unexposed_value, nrow=self.__nrow, ncol=self.__ncol, use_zero_storage=self.__use_zero_storage, use_symmetric_storage=self.__use_symmetric_storage)

        val = <COMPLEX64_t *> PyMem_Malloc(nnz * sizeof(COMPLEX64_t))
        if not val:
            raise MemoryError()
        memcpy(val, self.val, nnz * sizeof(COMPLEX64_t))
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

        self_copy.__col_indices_sorted_test_done = self.__col_indices_sorted_test_done
        self_copy.__col_indices_sorted = self.__col_indices_sorted
        self_copy.__first_row_not_ordered = self.__first_row_not_ordered

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
        if self.__use_symmetric_storage:
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
                    elif real_j < self.col[k]:
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
                    elif j < self.col[k]:
                        break

            else:
                for k from self.ind[i] <= k < self.ind[i+1]:
                    if j == self.col[k]:
                        return self.val[k]

        return 0.0

    # EXPLICIT TYPE TESTS

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX64_t safe_at(self, INT32_t i, INT32_t j) except *:

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

    ####################################################################################################################
    # Common operations
    ####################################################################################################################
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
            cnp.ndarray[cnp.npy_complex64, ndim=1] a_val = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_COMPLEX64)

            # Intermediate pointers to matrix data
            INT32_t   *pi
            INT32_t   *pj
            COMPLEX64_t    *pv
            INT32_t   i, k, elem

        pi = <INT32_t *> cnp.PyArray_DATA(a_row)
        pj = <INT32_t *> cnp.PyArray_DATA(a_col)
        pv = <COMPLEX64_t *> cnp.PyArray_DATA(a_val)

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
            cnp.ndarray[cnp.npy_complex64, ndim=1] diag = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_COMPLEX64)
            COMPLEX64_t    *pv
            INT32_t   i, k_

        pv = <COMPLEX64_t *> cnp.PyArray_DATA(diag)

        # init NumPy array
        for i from 0 <= i < diag_size:

            pv[i] = 0.0 + 0.0j


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

    def tril(self, int k):
        """
        Return the lower triangular part of the matrix.

        Args:
            k: (k<=0) the last diagonal to be included in the lower triangular part.

        Returns:
            A ``CSRSparseMatrix`` with the lower triangular part.

        Raises:
            IndexError if the diagonal number is out of bounds.

        """
        if k > 0:
            raise IndexError("k-th diagonal must be <= 0 (here: k = %d)" % k)

        if k < -self.nrow + 1:
            raise IndexError("k_th diagonal must be %d <= k <= 0 (here: k = %d)" % (-self.nrow + 1, k))

        # create internal arrays (big enough to contain all elements)

        cdef INT32_t * ind = <INT32_t *> PyMem_Malloc((self.__nrow + 1) * sizeof(INT32_t))
        if not ind:
            raise MemoryError()

        cdef INT32_t * col = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
        if not col:
            PyMem_Free(ind)
            raise MemoryError()

        cdef COMPLEX64_t * val = <COMPLEX64_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX64_t))
        if not val:
            PyMem_Free(ind)
            PyMem_Free(col)
            raise MemoryError()

        # populate arrays
        cdef:
            INT32_t i, j, k_, nnz

        nnz = 0
        ind[0] = 0

        for i from 0 <= i < self.__nrow:
            for k_ from self.ind[i] <= k_ < self.ind[i+1]:
                j = self.col[k_]
                v = self.val[k_]

                if i >= j - k:
                    col[nnz] = j
                    val[nnz] = v
                    nnz += 1

            ind[i+1] = nnz

        # resize arrays col and val
        cdef:
            void *temp

        temp = <INT32_t *> PyMem_Realloc(col, nnz * sizeof(INT32_t))
        col = <INT32_t*>temp

        temp = <COMPLEX64_t *> PyMem_Realloc(val, nnz * sizeof(COMPLEX64_t))
        val = <COMPLEX64_t*>temp

        return MakeCSRSparseMatrix_INT32_t_COMPLEX64_t(self.__nrow,
                                                  self.__ncol,
                                                  nnz,
                                                  ind,
                                                  col,
                                                  val,
                                                  use_symmetric_storage=False,
                                                  use_zero_storage=self.__use_zero_storage,
                                                  col_indices_are_sorted=True)

    def triu(self, int k):
        """
        Return the upper triangular part of the matrix.

        Args:
            k: (k>=0) the last diagonal to be included in the upper triangular part.

        Returns:
            A ``CSRSparseMatrix`` with the upper triangular part.

        Raises:
            IndexError if the diagonal number is out of bounds.

        """
        if k < 0:
            raise IndexError("k-th diagonal must be >= 0 (here: k = %d)" % k)

        if k > self.ncol - 1:
            raise IndexError("k_th diagonal must be 0 <= k <= %d (here: k = %d)" % (-self.ncol - 1, k))

        # create internal arrays (big enough to contain all elements)

        cdef INT32_t * ind = <INT32_t *> PyMem_Malloc((self.__nrow + 1) * sizeof(INT32_t))
        if not ind:
            raise MemoryError()

        cdef INT32_t * col = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
        if not col:
            PyMem_Free(ind)
            raise MemoryError()

        cdef COMPLEX64_t * val = <COMPLEX64_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX64_t))
        if not val:
            PyMem_Free(ind)
            PyMem_Free(col)
            raise MemoryError()

        # populate arrays
        cdef:
            INT32_t i, j, k_, nnz

        nnz = 0
        ind[0] = 0

        # Special case: when matrix is symmetric: we first create an internal CSC and then translate it to CSR
        cdef INT32_t * csc_ind
        cdef INT32_t * csc_row
        cdef COMPLEX64_t  * csc_val

        if self.__use_symmetric_storage:
            # Special (and annoying) case: we first create a CSC and then translate it to CSR
            csc_ind = <INT32_t *> PyMem_Malloc((self.__ncol + 1) * sizeof(INT32_t))
            if not csc_ind:
                PyMem_Free(ind)
                PyMem_Free(col)
                PyMem_Free(val)

                raise MemoryError()

            csc_row = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
            if not csc_row:
                PyMem_Free(ind)
                PyMem_Free(col)
                PyMem_Free(val)

                PyMem_Free(csc_ind)
                raise MemoryError()

            csc_val = <COMPLEX64_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX64_t))
            if not csc_val:
                PyMem_Free(ind)
                PyMem_Free(col)
                PyMem_Free(val)

                PyMem_Free(csc_ind)
                PyMem_Free(csc_row)
                raise MemoryError()

            csc_ind[0] = 0

            for i from 0 <= i < self.__nrow:
                for k_ from self.ind[i] <= k_ < self.ind[i+1]:
                    j = self.col[k_]
                    v = self.val[k_]

                    if i >= j + k:
                        csc_row[nnz] = j
                        csc_val[nnz] = v
                        nnz += 1

                csc_ind[i+1] = nnz

            csc_to_csr_kernel_INT32_t_COMPLEX64_t(self.__nrow, self.__ncol, nnz,
                                      csc_ind, csc_row, csc_val,
                                      ind, col, val)

            # erase temp arrays
            PyMem_Free(csc_ind)
            PyMem_Free(csc_row)
            PyMem_Free(csc_val)

        else:  # not symmetric
            for i from 0 <= i < self.__nrow:
                for k_ from self.ind[i] <= k_ < self.ind[i+1]:
                    j = self.col[k_]
                    v = self.val[k_]

                    if i <= j - k:
                        col[nnz] = j
                        val[nnz] = v
                        nnz += 1

                ind[i+1] = nnz

        # resize arrays col and val
        cdef:
            void *temp

        temp = <INT32_t *> PyMem_Realloc(col, nnz * sizeof(INT32_t))
        col = <INT32_t*>temp

        temp = <COMPLEX64_t *> PyMem_Realloc(val, nnz * sizeof(COMPLEX64_t))
        val = <COMPLEX64_t*>temp

        return MakeCSRSparseMatrix_INT32_t_COMPLEX64_t(self.__nrow,
                                                  self.__ncol,
                                                  nnz,
                                                  ind,
                                                  col,
                                                  val,
                                                  use_symmetric_storage=False,
                                                  use_zero_storage=self.__use_zero_storage,
                                                  col_indices_are_sorted=True)

    def to_csc(self):
        """
        Transform this matrix into a :class:`CSRSparseMatrix`.

        """

        # create CSC internal arrays: ind, row and val
        cdef INT32_t * ind = <INT32_t *> PyMem_Malloc((self.__ncol + 1) * sizeof(INT32_t))
        if not ind:
            raise MemoryError()

        cdef INT32_t * row = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
        if not row:
            PyMem_Free(ind)
            raise MemoryError()

        cdef COMPLEX64_t * val = <COMPLEX64_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX64_t))
        if not val:
            PyMem_Free(ind)
            PyMem_Free(row)
            raise MemoryError()

        csr_to_csc_kernel_INT32_t_COMPLEX64_t(self.__nrow, self.__ncol, self.__nnz,
                       <INT32_t *>self.ind, <INT32_t *>self.col, <COMPLEX64_t *>self.val,
                       ind, row, val)

        return MakeCSCSparseMatrix_INT32_t_COMPLEX64_t(self.__nrow,
                                                  self.__ncol,
                                                  self.__nnz,
                                                  ind,
                                                  row,
                                                  val,
                                                  use_symmetric_storage=self.use_symmetric_storage,
                                                  use_zero_storage=self.use_zero_storage,
                                                  row_indices_are_sorted=True)


    def to_ndarray(self):
        """
        Return the matrix in the form of a :program:`NumPy` ``ndarray``.

        """
        # EXPLICIT TYPE TESTS
        cdef:
            cnp.ndarray[cnp.npy_complex64, ndim=2] np_ndarray
            INT32_t i, j, k
            COMPLEX64_t [:,:] np_memview
            COMPLEX64_t value

        np_ndarray = np.zeros((self.__nrow, self.__ncol), dtype=np.complex64, order='C')
        np_memview = np_ndarray

        if not self.__use_symmetric_storage:
            for i from 0 <= i < self.__nrow:
                for k from self.ind[i] <= k < self.ind[i+1]:
                    np_memview[i, self.col[k]] = self.val[k]

        else:
            for i from 0 <= i < self.__nrow:
                for k from self.ind[i] <= k < self.ind[i+1]:
                    j = self.col[k]
                    value = self.val[k]

                    np_memview[i, j] = value
                    np_memview[j, i] = value

        return np_ndarray

    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def matvec(self, b):
        """
        Return :math:`A * b`.
        """
        return multiply_csr_mat_with_numpy_vector_INT32_t_COMPLEX64_t(self, b)

    def matvec_transp(self, b):
        """
        Return :math:`A^t * b`.
        """
        return multiply_transposed_csr_mat_with_numpy_vector_INT32_t_COMPLEX64_t(self, b)


    def matvec_htransp(self, b):
        """
        Return :math:`A^h * b`.
        """
        assert are_mixed_types_compatible(COMPLEX64_T, b.dtype), "Multiplication only allowed with a Numpy compatible type (%s)!" % cysparse_to_numpy_type(COMPLEX64_T)
        return multiply_conjugate_transposed_csr_mat_with_numpy_vector_INT32_t_COMPLEX64_t(self, b)

    def matvec_conj(self, b):
        """
        Return :math:`\textrm{conj}(A) * b`.
        """
        assert are_mixed_types_compatible(COMPLEX64_T, b.dtype), "Multiplication only allowed with a Numpy compatible type (%s)!" % cysparse_to_numpy_type(COMPLEX64_T)
        return multiply_conjugated_csr_mat_with_numpy_vector_INT32_t_COMPLEX64_t(self, b)


    def matdot(self, B):
        raise NotImplementedError("Multiplication with this kind of object not allowed")

    def matdot_transp(self, B):
        raise NotImplementedError("Multiplication with this kind of object not allowed")

    def __mul__(self, other):

        # test if implemented
        if isinstance(other, (MutableSparseMatrix_INT32_t_COMPLEX64_t, ImmutableSparseMatrix_INT32_t_COMPLEX64_t)):
            pass
        else:
            raise NotImplemented("Multiplication not (yet) allowed")

        # CASES
        if isinstance(other, CSCSparseMatrix_INT32_t_COMPLEX64_t):
            return multiply_csr_mat_by_csc_mat_INT32_t_COMPLEX64_t(self, other)
        else:
            raise NotImplemented("Multiplication not (yet) allowed")

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def at_to_string(self, INT32_t i, INT32_t j, int cell_width=10):
        """
        Return a string with a given element if it exists or an "empty" string.


        """
        cdef:
            INT32_t k

        for k from self.ind[i] <= k < self.ind[i+1]:
            if j == self.col[k]:
                return element_to_string_COMPLEX64_t(self.val[k], cell_width=cell_width)

        # element not found -> return empty cell
        return empty_to_string_COMPLEX64_t(cell_width=cell_width)

    def at_conj_to_string(self, INT32_t i, INT32_t j, int cell_width=10):
        """
        Return a string with a given element if it exists or an "empty" string.


        """
        cdef:
            INT32_t k

        for k from self.ind[i] <= k < self.ind[i+1]:
            if j == self.col[k]:
                return conjugated_element_to_string_COMPLEX64_t(self.val[k], cell_width=cell_width)

        # element not found -> return empty cell
        return empty_to_string_COMPLEX64_t(cell_width=cell_width)

    #def __repr__(self):
    #    s = "CSRSparseMatrix of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
    #    return s

    def print_to(self, OUT, width=9, print_big_matrices=False, transposed=False):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.
        """
        # EXPLICIT TYPE TESTS
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        cdef INT32_t i, k, first = 1;

        cdef COMPLEX64_t *mat
        cdef INT32_t j
        cdef COMPLEX64_t val

        print(self._matrix_description_before_printing(), file=OUT)
        #print('CSRSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz:
            return

        if self.nrow <= CSR_MAT_PPRINT_COL_THRESH and self.ncol <= CSR_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: put in a method of its own
            mat = <COMPLEX64_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(COMPLEX64_t))

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
    # Internal arrays
    ####################################################################################################################
    # TODO: test, test, test!
    def get_c_pointers(self):
        """
        Return C pointers to internal arrays.

        Returns:
            Triple `(ind, col, val)`.

        Warning:
            The returned values can only be used by C-extensions.
        """
        cdef:
            PyObject * ind_obj = <PyObject *> self.ind
            PyObject * col_obj = <PyObject *> self.col
            PyObject * val_obj = <PyObject *> self.val

        return <object>ind_obj, <object>col_obj, <object>val_obj

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
        dim[0] = self.nrow + 1
        ind_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_INT32, <INT32_t *>self.ind)

        # col
        dim[0] = self.nnz
        col_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_INT32, <INT32_t *>self.col)

        # val
        dim[0] = self.nnz
        val_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_COMPLEX64, <COMPLEX64_t *>self.val)


        return ind_numpy_array, col_numpy_array, val_numpy_array

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



    def set_col(self, INT32_t i, INT32_t val):
        self.col[i] = val

########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSRSparseMatrix_INT32_t_COMPLEX64_t(INT32_t nrow,
                                        INT32_t ncol,
                                        INT32_t nnz,
                                        INT32_t * ind,
                                        INT32_t * col,
                                        COMPLEX64_t * val,
                                        bint use_symmetric_storage,
                                        bint use_zero_storage,
                                        bint col_indices_are_sorted=False):
    """
    Construct a CSRSparseMatrix object.

    Args:
        nrow (INT32_t): Number of rows.
        ncol (INT32_t): Number of columns.
        nnz (INT32_t): Number of non-zeros.
        ind (INT32_t *): C-array with column indices pointers.
        col  (INT32_t *): C-array with column indices.
        val  (COMPLEX64_t *): C-array with values.
        use_symmetric_storage (boolean): Is matrix symmetrix or not?
        use_zero_storage (boolean): Do we store zeros or not?
        col_indices_are_sorted (boolean): Are the column indices sorted or not?
    """
    cdef CSRSparseMatrix_INT32_t_COMPLEX64_t csr_mat

    csr_mat = CSRSparseMatrix_INT32_t_COMPLEX64_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz, use_symmetric_storage=use_symmetric_storage, use_zero_storage=use_zero_storage)

    csr_mat.val = val
    csr_mat.ind = ind
    csr_mat.col = col

    if col_indices_are_sorted:
        csr_mat._set_column_indices_ordered_is_true()

    return csr_mat
