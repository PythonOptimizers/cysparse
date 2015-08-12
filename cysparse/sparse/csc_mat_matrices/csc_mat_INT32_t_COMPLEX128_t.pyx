"""
Condensed Sparse Column (CSC) Format Matrices.


"""
from __future__ import print_function

from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat cimport unexposed_value

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_COMPLEX128_t cimport ImmutableSparseMatrix_INT32_t_COMPLEX128_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX128_t cimport LLSparseMatrix_INT32_t_COMPLEX128_t

from cysparse.sparse.sparse_utils.generic.sort_indices_INT32_t cimport sort_array_INT32_t
from cysparse.sparse.sparse_utils.generic.print_COMPLEX128_t cimport element_to_string_COMPLEX128_t, conjugated_element_to_string_COMPLEX128_t, empty_to_string_COMPLEX128_t
from cysparse.sparse.sparse_utils.generic.matrix_translations_INT32_t_COMPLEX128_t cimport csr_to_csc_kernel_INT32_t_COMPLEX128_t, csc_to_csr_kernel_INT32_t_COMPLEX128_t

from cysparse.sparse.csr_mat_matrices.csr_mat_INT32_t_COMPLEX128_t cimport CSRSparseMatrix_INT32_t_COMPLEX128_t, MakeCSRSparseMatrix_INT32_t_COMPLEX128_t

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

include "csc_mat_kernel/csc_mat_multiplication_by_numpy_vector_kernel_INT32_t_COMPLEX128_t.pxi"
include "csc_mat_helpers/csc_mat_multiplication_INT32_t_COMPLEX128_t.pxi"


cdef class CSCSparseMatrix_INT32_t_COMPLEX128_t(ImmutableSparseMatrix_INT32_t_COMPLEX128_t):
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

        self.__row_indices_sorted_test_done = False
        self.__row_indices_sorted = False
        self.__first_col_not_ordered = -1

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

        cdef CSCSparseMatrix_INT32_t_COMPLEX128_t self_copy

        # we copy manually the C-arrays
        cdef:
            COMPLEX128_t * val
            INT32_t * row
            INT32_t * ind
            INT32_t nnz

        nnz = self.nnz

        self_copy = CSCSparseMatrix_INT32_t_COMPLEX128_t(control_object=unexposed_value, nrow=self.__nrow, ncol=self.__ncol, store_zeros=self.__store_zeros, is_symmetric=self.__is_symmetric)

        val = <COMPLEX128_t *> PyMem_Malloc(nnz * sizeof(COMPLEX128_t))
        if not val:
            raise MemoryError()
        memcpy(val, self.val, nnz * sizeof(COMPLEX128_t))
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

        self_copy.__row_indices_sorted_test_done = self.__row_indices_sorted_test_done
        self_copy.__row_indices_sorted = self.__row_indices_sorted
        self_copy.__first_col_not_ordered = self.__first_col_not_ordered

        return self_copy

    ####################################################################################################################
    # Row indices ordering
    ####################################################################################################################
    def are_row_indices_sorted(self, force_test=False):
        """
        Tell if row indices are sorted in augmenting order (ordered).


        """
        cdef INT32_t j
        cdef INT32_t row_index
        cdef INT32_t row_index_stop

        if not force_test and self.__row_indices_sorted_test_done:
            return self.__row_indices_sorted
        else:
            # do the test
            self.__row_indices_sorted_test_done = True
            # test each row
            for j from 0 <= j < self.__ncol:
                row_index = self.ind[j]
                row_index_stop = self.ind[j+1] - 1

                self.__first_col_not_ordered = j

                while row_index < row_index_stop:
                    if self.row[row_index] > self.row[row_index + 1]:
                        self.__row_indices_sorted = False
                        return self.__row_indices_sorted
                    row_index += 1

        # row indices are ordered
        self.__first_col_not_ordered = self.__ncol
        self.__row_indices_sorted = True
        return self.__row_indices_sorted

    cdef _set_row_indices_ordered_is_true(self):
        """
        If you construct a CSC matrix and you know that its row indices **are** ordered, confirm it by calling this method.

        Warning:
            Be sure to know what you are doing because there is no control and we assume that the row indices are indeed sorted for
            almost all operations.
        """
        self.__row_indices_sorted_test_done = True
        self.__row_indices_sorted = True

    cdef _order_row_indices(self):
        """
        Order row indices by ascending order.

        We use a simple insert sort. The idea is that the row indices aren't that much not ordered.
        """
        #  must be called to find first col not ordered
        if self.are_row_indices_sorted():
            return

        cdef INT32_t j = self.__first_col_not_ordered
        cdef INT32_t row_index
        cdef INT32_t row_index_start
        cdef INT32_t row_index_stop

        while j < self.__ncol:
            row_index = self.ind[j]
            row_index_start = row_index
            row_index_stop = self.ind[j+1]

            while row_index < row_index_stop - 1:
                # detect if col is not ordered
                if self.row[row_index] > self.row[row_index + 1]:
                    # sort
                    # TODO: maybe use the row index for optimization?
                    sort_array_INT32_t(self.row, row_index_start, row_index_stop)
                    break
                else:
                    row_index += 1

            j += 1

    def order_row_indices(self):
        """
        Forces row indices to be ordered.
        """
        return self._order_row_indices()

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
            if i < j:
                real_i = j
                real_j = i
            else:
                real_i = i
                real_j = j

            if self. __row_indices_sorted:
                for k from self.ind[real_j] <= k < self.ind[real_j+1]:
                    if real_i == self.row[k]:
                        return self.val[k]
                    elif real_i < self.row[k]:
                        break
            else:
                for k from self.ind[real_j] <= k < self.ind[real_j+1]:
                    if real_i == self.row[k]:
                        return self.val[k]

        else:  # not symmetric
            if self. __row_indices_sorted:
                for k from self.ind[j] <= k < self.ind[j+1]:
                    if i == self.row[k]:
                        return self.val[k]
                    elif i < self.row[k]:
                        break
            else:
                for k from self.ind[j] <= k < self.ind[j+1]:
                    if i == self.row[k]:
                        return self.val[k]

        return 0.0

    # EXPLICIT TYPE TESTS

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX128_t safe_at(self, INT32_t i, INT32_t j) except *:

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

    ####################################################################################################################
    # Common operations
    ####################################################################################################################
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
            cnp.ndarray[cnp.npy_complex128, ndim=1] a_val = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_COMPLEX128)

            INT32_t   *pi
            INT32_t   *pj
            COMPLEX128_t    *pv
            INT32_t   j, k, elem

        pi = <INT32_t *> cnp.PyArray_DATA(a_row)
        pj = <INT32_t *> cnp.PyArray_DATA(a_col)
        pv = <COMPLEX128_t *> cnp.PyArray_DATA(a_val)

        elem = 0
        for j from 0 <= j < self.__ncol:
            for k from self.ind[j] <= k < self.ind[j+1]:
                pi[ elem ] = self.row[j]
                pj[ elem ] = j
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
            cnp.ndarray[cnp.npy_complex128, ndim=1] diag = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_COMPLEX128)
            COMPLEX128_t    *pv
            INT32_t   i, j, k_

        pv = <COMPLEX128_t *> cnp.PyArray_DATA(diag)

        # init NumPy array
        for i from 0 <= i < diag_size:

            pv[i] = 0.0 + 0.0j


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

    def tril(self, int k):
        """
        Return the lower triangular part of the matrix.

        Args:
            k: (k<=0) the last diagonal to be included in the lower triangular part.

        Returns:
            A ``CSCSparseMatrix`` with the lower triangular part.

        Raises:
            IndexError if the diagonal number is out of bounds.

        """
        if k > 0:
            raise IndexError("k-th diagonal must be <= 0 (here: k = %d)" % k)

        if k < -self.nrow + 1:
            raise IndexError("k_th diagonal must be %d <= k <= 0 (here: k = %d)" % (-self.nrow + 1, k))

        # create internal arrays (big enough to contain all elements)

        cdef INT32_t * ind = <INT32_t *> PyMem_Malloc((self.__ncol + 1) * sizeof(INT32_t))
        if not ind:
            raise MemoryError()

        cdef INT32_t * row = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
        if not row:
            PyMem_Free(ind)
            raise MemoryError()

        cdef COMPLEX128_t * val = <COMPLEX128_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX128_t))
        if not val:
            PyMem_Free(ind)
            PyMem_Free(row)
            raise MemoryError()

        # populate arrays
        cdef:
            INT32_t i, j, k_, nnz

        nnz = 0
        ind[0] = 0

        for j from 0 <= j < self.__ncol:
            for k_ from self.ind[j] <= k_ < self.ind[j+1]:
                i = self.row[k_]
                v = self.val[k_]

                if i >= j - k:
                    row[nnz] = i
                    val[nnz] = v
                    nnz += 1

            ind[j+1] = nnz

        # resize arrays row and val
        cdef:
            void *temp

        temp = <INT32_t *> PyMem_Realloc(row, nnz * sizeof(INT32_t))
        row = <INT32_t*>temp

        temp = <COMPLEX128_t *> PyMem_Realloc(val, nnz * sizeof(COMPLEX128_t))
        val = <COMPLEX128_t*>temp

        return MakeCSCSparseMatrix_INT32_t_COMPLEX128_t(self.__nrow,
                                                  self.__ncol,
                                                  nnz,
                                                  ind,
                                                  row,
                                                  val,
                                                  is_symmetric=False,
                                                  store_zeros=self.__store_zeros,
                                                  row_indices_are_sorted==True)

    def triu(self, int k):
        """
        Return the upper triangular part of the matrix.

        Args:
            k: (k>=0) the last diagonal to be included in the upper triangular part.

        Returns:
            A ``CSCSparseMatrix`` with the upper triangular part.

        Raises:
            IndexError if the diagonal number is out of bounds.

        """
        if k < 0:
            raise IndexError("k-th diagonal must be >= 0 (here: k = %d)" % k)

        if k > self.ncol - 1:
            raise IndexError("k_th diagonal must be 0 <= k <= %d (here: k = %d)" % (-self.ncol - 1, k))

        # create internal arrays (big enough to contain all elements)

        cdef INT32_t * ind = <INT32_t *> PyMem_Malloc((self.__ncol + 1) * sizeof(INT32_t))
        if not ind:
            raise MemoryError()

        cdef INT32_t * row = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
        if not row:
            PyMem_Free(ind)
            raise MemoryError()

        cdef COMPLEX128_t * val = <COMPLEX128_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX128_t))
        if not val:
            PyMem_Free(ind)
            PyMem_Free(row)
            raise MemoryError()

        # populate arrays
        cdef:
            INT32_t i, j, k_, nnz

        nnz = 0
        ind[0] = 0

        # Special case: when matrix is symmetric: we first create an internal CSR and then translate it to CSC
        cdef INT32_t * csr_ind
        cdef INT32_t * csr_col
        cdef COMPLEX128_t  * csr_val

        if self.__is_symmetric:
            # Special (and annoying) case: we first create a CSR and then translate it to CSC
            csr_ind = <INT32_t *> PyMem_Malloc((self.__nrow + 1) * sizeof(INT32_t))
            if not csr_ind:
                PyMem_Free(ind)
                PyMem_Free(row)
                PyMem_Free(val)

                raise MemoryError()

            csr_col = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
            if not csr_col:
                PyMem_Free(ind)
                PyMem_Free(row)
                PyMem_Free(val)

                PyMem_Free(csr_ind)
                raise MemoryError()

            csr_val = <COMPLEX128_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX128_t))
            if not csr_val:
                PyMem_Free(ind)
                PyMem_Free(row)
                PyMem_Free(val)

                PyMem_Free(csr_ind)
                PyMem_Free(csr_col)
                raise MemoryError()

            csr_ind[0] = 0

            for j from 0 <= j < self.__ncol:
                for k_ from self.ind[j] <= k_ < self.ind[j+1]:
                    i = self.row[k_]
                    v = self.val[k_]

                    if i >= j + k:
                        csr_col[nnz] = i
                        csr_val[nnz] = v
                        nnz += 1

                csr_ind[j+1] = nnz

            csr_to_csc_kernel_INT32_t_COMPLEX128_t(self.__nrow, self.__ncol, nnz,
                                      csr_ind, csr_col, csr_val,
                                      ind, row, val)

            # erase temp arrays
            PyMem_Free(csr_ind)
            PyMem_Free(csr_col)
            PyMem_Free(csr_val)

        else:  # not symmetric
            for j from 0 <= j < self.__ncol:
                for k_ from self.ind[j] <= k_ < self.ind[j+1]:
                    i = self.row[k_]
                    v = self.val[k_]

                    if i <= j - k:
                        row[nnz] = i
                        val[nnz] = v
                        nnz += 1

                ind[j+1] = nnz

        # resize arrays row and val
        cdef:
            void *temp

        temp = <INT32_t *> PyMem_Realloc(row, nnz * sizeof(INT32_t))
        row = <INT32_t*>temp

        temp = <COMPLEX128_t *> PyMem_Realloc(val, nnz * sizeof(COMPLEX128_t))
        val = <COMPLEX128_t*>temp

        return MakeCSCSparseMatrix_INT32_t_COMPLEX128_t(self.__nrow,
                                                  self.__ncol,
                                                  nnz,
                                                  ind,
                                                  row,
                                                  val,
                                                  is_symmetric=False,
                                                  store_zeros=self.__store_zeros,
                                                  row_indices_are_sorted==True)

    def to_csr(self):
        """
        Transform this matrix into a :class:`CSRSparseMatrix`.

        """
        # create CSR internal arrays: ind, col and val
        cdef INT32_t * ind = <INT32_t *> PyMem_Malloc((self.__nrow + 1) * sizeof(INT32_t))
        if not ind:
            raise MemoryError()

        cdef INT32_t * col = <INT32_t *> PyMem_Malloc(self.__nnz * sizeof(INT32_t))
        if not col:
            PyMem_Free(ind)
            raise MemoryError()

        cdef COMPLEX128_t * val = <COMPLEX128_t *> PyMem_Malloc(self.__nnz * sizeof(COMPLEX128_t))
        if not val:
            PyMem_Free(ind)
            PyMem_Free(col)
            raise MemoryError()

        csc_to_csr_kernel_INT32_t_COMPLEX128_t(self.__nrow, self.__ncol, self.__nnz,
                       <INT32_t *>self.ind, <INT32_t *>self.row, <COMPLEX128_t *>self.val,
                       ind, col, val)

        return MakeCSRSparseMatrix_INT32_t_COMPLEX128_t(self.__nrow,
                                                  self.__ncol,
                                                  self.__nnz,
                                                  ind,
                                                  col,
                                                  val,
                                                  is_symmetric=self.is_symmetric,
                                                  store_zeros=self.store_zeros,
                                                  row_indices_are_sorted=True)


    def to_ndarray(self):
        """
        Return the matrix in the form of a :program:`NumPy` ``ndarray``.

        """
        # EXPLICIT TYPE TESTS
        cdef:
            cnp.ndarray[cnp.npy_complex128, ndim=2] np_ndarray
            INT32_t i, j, k
            COMPLEX128_t [:,:] np_memview

        np_ndarray = np.zeros((self.__nrow, self.__ncol), dtype=np.complex128, order='C')
        np_memview = np_ndarray

        for j from 0 <= j < self.__ncol:
            for k from self.ind[j] <= k < self.ind[j+1]:
                np_memview[self.row[k], j] = self.val[k]

        return np_ndarray

    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def matvec(self, b):
        """
        Return :math:`A * b`.
        """
        return multiply_csc_mat_with_numpy_vector_INT32_t_COMPLEX128_t(self, b)

    def matvec_transp(self, b):
        """
        Return :math:`A^t * b`.
        """
        return multiply_transposed_csc_mat_with_numpy_vector_INT32_t_COMPLEX128_t(self, b)


    def matvec_htransp(self, b):
        """
        Return :math:`A^h * b`.
        """
        return multiply_conjugate_transposed_csc_mat_with_numpy_vector_INT32_t_COMPLEX128_t(self, b)

    def matvec_conj(self, b):
        """
        Return :math:`\textrm{conj}(A) * b`.
        """
        return multiply_conjugated_csc_mat_with_numpy_vector_INT32_t_COMPLEX128_t(self, b)


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
    def at_to_string(self, INT32_t i, INT32_t j, int cell_width=10):
        """
        Return a string with a given element if it exists or an "empty" string.


        """
        cdef:
            INT32_t k

        for k from self.ind[j] <= k < self.ind[j+1]:
            if i == self.row[k]:
                return element_to_string_COMPLEX128_t(self.val[k], cell_width=cell_width)

        # element not found -> return empty cell
        return empty_to_string_COMPLEX128_t(cell_width=cell_width)

    def at_conj_to_string(self, INT32_t i, INT32_t j, int cell_width=10):
        """
        Return a string with a given element if it exists or an "empty" string.


        """
        cdef:
            INT32_t k

        for k from self.ind[j] <= k < self.ind[j+1]:
            if i == self.row[k]:
                return conjugated_element_to_string_COMPLEX128_t(self.val[k], cell_width=cell_width)

        # element not found -> return empty cell
        return empty_to_string_COMPLEX128_t(cell_width=cell_width)

    #def __repr__(self):
    #    s = "CSCSparseMatrix of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
    #    return s

    def print_to(self, OUT):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.
        """
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        cdef INT32_t i, k, first = 1;

        cdef COMPLEX128_t *mat
        cdef INT32_t j
        cdef COMPLEX128_t val

        print('CSCSparseMatrix ([%d,%d]):' % (self.nrow, self.ncol), file=OUT)

        if not self.nnz:
            return

        if self.nrow <= CSC_MAT_PPRINT_COL_THRESH and self.ncol <= CSC_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: Skip this creation
            mat = <COMPLEX128_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(COMPLEX128_t))

            if not mat:
                raise MemoryError()

            for j from 0 <= j < self.ncol:
                for i from 0 <= i < self.nrow:


                    mat[i* self.ncol + j] = 0.0 + 0.0j


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
    # Internal arrays
    ####################################################################################################################
    # TODO: test, test, test!
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
        val_numpy_array = cnp.PyArray_SimpleNewFromData(1, dim, cnp.NPY_COMPLEX128, <COMPLEX128_t *>self.val)


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
cdef MakeCSCSparseMatrix_INT32_t_COMPLEX128_t(INT32_t nrow,
                                        INT32_t ncol,
                                        INT32_t nnz,
                                        INT32_t * ind,
                                        INT32_t * row,
                                        COMPLEX128_t * val,
                                        bint is_symmetric, bint store_zeros,
                                        bint row_indices_are_sorted=False):
    """
    Construct a CSCSparseMatrix object.

    Args:
        nrow (INT32_t): Number of rows.
        ncol (INT32_t): Number of columns.
        nnz (INT32_t): Number of non-zeros.
        ind (INT32_t *): C-array with column indices pointers.
        row  (INT32_t *): C-array with row indices.
        val  (COMPLEX128_t *): C-array with values.
    """
    cdef CSCSparseMatrix_INT32_t_COMPLEX128_t csc_mat
    csc_mat = CSCSparseMatrix_INT32_t_COMPLEX128_t(control_object=unexposed_value,
                                             nrow=nrow,
                                             ncol=ncol,
                                             nnz=nnz,
                                             is_symmetric=is_symmetric,
                                             store_zeros=store_zeros)

    csc_mat.val = val
    csc_mat.ind = ind
    csc_mat.row = row

    if row_indices_are_sorted:
        csc_mat._set_row_indices_ordered_is_true()

    return csc_mat
