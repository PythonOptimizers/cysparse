"""
Lightweight object to view a :class:`LLSparseMatrix`.


"""

# forward declaration
cdef class LLSparseMatrixView

from sparse_lib.sparse.ll_mat cimport LLSparseMatrix
from sparse_lib.utils.equality cimport values_are_equal

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject

cimport numpy as cnp
cnp.import_array()

import numpy as np

include "object_index.pxi"

cdef class LLSparseMatrixView:
    def __cinit__(self, LLSparseMatrix A, int nrow, int ncol):
        self.nrow = nrow
        self.ncol = ncol

        self.is_empty = True

        self.A = A

        self.is_symmetric = A.is_symmetric
        self.store_zeros = A.store_zeros

        self.__status_ok = False
        self.__counted_nnz = False

    property nnz:
        # we only count once the non zero elements
        # TODO: take symmetry into account
        def __get__(self):
            if not self.__counted_nnz:
                # we have to count the nnz
                self._nnz = self.count_nnz()

            return self._nnz

        def __set__(self, value):
            raise NotImplemented("nnz is read-only")

        def __del__(self):
            raise NotImplemented("nnz is read-only")

    def __dealloc__(self):
        PyMem_Free(self.row_indices)
        PyMem_Free(self.col_indices)

    cdef assert_status_ok(self):
        assert self.__status_ok, "Create an LLSparseMatrixView only with the factory function MakeLLSparseMatrixView()"

    def __setitem__(self, tuple key, value):
        # TODO: direct access to the matrix
        raise NotImplemented("This operation is not allowed for LLSparseMatrixView")

    def __getitem__(self, tuple):
        # TODO: return another view
        raise NotImplemented("This operation is not allowed for LLSparseMatrixView")

    def copy(self, compress=True):
        """
        Create a new :class:`LLSparseMatrix` from the view and return it.

        Args:
            compress: If ``True``, we use the minimum size for the matrix.

        Note:
            It doesn't make sense to construct a copy of a view as the view cannot be altered. It's only the viewed
            matrix that is altered, **not** a view to it. So, in a sense, a `LLSparseMatrixView` object is immutable.

        """
        self.assert_status_ok()

        cdef int size_hint
        cdef double val

        if compress:
            size_hint = self.count_nnz()
        else:
            size_hint = min(self.nrow * self.ncol, self.A.nalloc)

        cdef LLSparseMatrix A_copy = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, size_hint=size_hint, store_zeros=self.store_zeros)

        cdef:
            int i
            int row_index

        # TODO: is this the right thing to do (about store_zeros)?
        # TODO: take values immediately with at(i, j)...
        if self.store_zeros:
            for i from 0 <= i < self.nrow:
                row_index = self.row_indices[i]
                for j from 0 <= j < self.ncol:
                    A_copy[i, j] = self.A[row_index, self.col_indices[j]]

        else:
            for i from 0 <= i < self.nrow:
                row_index = self.row_indices[i]
                for j from 0 <= j < self.ncol:
                    val = self.A[row_index, self.col_indices[j]]
                    if not values_are_equal(val, 0.0):
                        A_copy[i, j] = self.A[row_index, self.col_indices[j]]

        return A_copy


    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def __mul__(self, B):
        # TODO: optimize all this: this implementation is horrible!
        # TODO: find common code to refactor to apply the DRY principle
        return self.copy() * B  # <-- Both (A either LLSparseMatrix or LLSparseMatrixView) A * B should be implemented with the same common function

        ## CASES
        #if isinstance(B, LLSparseMatrix):
        #    return self.copy() * B

        #elif isinstance(B, np.ndarray):
        #    return self.copy() * B


    cdef int count_nnz(self):
        """
        Count number of non zeros elements.

        Returns:
            The number of non zeros elements if the corresponding :class:`LLSparseMatrix` doesn't store zeros, otherwise
            returns the ``size = nrow * ncol``.
        """
        self.assert_status_ok()

        cdef:
            int i, j, row_index
            int nnz = 0

        if self.store_zeros:
            nnz = self.nrow * self.ncol
        else:
            for i from 0 <= i < self.nrow:
                row_index = self.row_indices[i]
                for j from 0 <= j < self.ncol:
                    if self.A[row_index, self.col_indices[j]] != 0.0:
                        nnz += 1

        return nnz


cdef LLSparseMatrixView MakeLLSparseMatrixView(LLSparseMatrix A, PyObject* obj1, PyObject* obj2):
    """
    Factory function to create a new :class:`LLSparseMatrixView` for a :class:`LLSparseMatrix`.

    Two index objects must be provided. Such objects can be:
        - an integer;
        - a list;
        - a slice;
        - a numpy array.

    Args:
        A: A :class:`LLSparseMatrix` to be *viewed*.
        obj1: First index object.
        obj2: Second index object.

    Raises:
        IndexError:
            - a variable in the index object is out of bound;
            - the dimension of a numpy array is not 1;
        RuntimeError:
            - a slice can not be interpreted;
        MemoryError:
            - there is not enough memory to translate an index object into a C-array of indices.

    Returns:
        A corresponding :class:`LLSparseMatrixView`. This view can be empty with the wrong index objects.

    Warning:
        This should be the only way to create a view to a :class:`LLSparseMatrix`.

    """
    cdef:
        int nrow
        int * row_indices,
        int ncol
        int * col_indices
        int A_nrow = A.nrow
        int A_ncol = A.ncol

    row_indices = create_c_array_indices_from_python_object(A_nrow, obj1, &nrow)
    col_indices = create_c_array_indices_from_python_object(A_ncol, obj2, &ncol)

    cdef LLSparseMatrixView view = LLSparseMatrixView(A, nrow, ncol)
    view.row_indices = row_indices
    view.col_indices = col_indices

    if nrow == 0 or ncol == 0:
        view.is_empty = True
    else:
        view.is_empty = False

    view.__status_ok = True

    return view


