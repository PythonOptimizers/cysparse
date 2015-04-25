"""
Lightweight object to view a :class:`LLSparseMatrix`.


"""
from sparse_lib.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView

from sparse_lib.sparse.ll_mat cimport LLSparseMatrix
from sparse_lib.utils.equality cimport values_are_equal

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject
from python_ref cimport Py_INCREF, Py_DECREF

cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)


include "indices/object_index.pxi"

cdef class LLSparseMatrixView:
    def __cinit__(self, LLSparseMatrix A, INT_t nrow, INT_t ncol):
        self.nrow = nrow  # number of rows of the view
        self.ncol = ncol  # number of columns of the view

        self.is_empty = True

        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.is_symmetric = A.is_symmetric
        self.store_zeros = A.store_zeros

        self.__status_ok = False
        self.__counted_nnz = False
        self._nnz = 0

    property nnz:
        # we only count once the non zero elements
        def __get__(self):
            if not self.__counted_nnz:
                # we have to count the nnz
                self._nnz = self.count_nnz()
                self.__counted_nnz = True

            return self._nnz

        def __set__(self, value):
            raise NotImplemented("nnz is read-only")

        def __del__(self):
            raise NotImplemented("nnz is read-only")

    def __dealloc__(self):
        PyMem_Free(self.row_indices)
        PyMem_Free(self.col_indices)

        Py_DECREF(self.A) # release ref

    cdef assert_status_ok(self):
        assert self.__status_ok, "Create an LLSparseMatrixView only with the factory function MakeLLSparseMatrixView()"

    ####################################################################################################################
    # Set/Get items
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    cdef put(self, INT_t i, INT_t j, double value):
        self.A.put(self.row_indices[i], self.col_indices[j], value)

    cdef safe_put(self, INT_t i, INT_t j, double value):
        """
        Set ``A_view[i, j] = value`` directly.

        Raises:
            IndexError: when index out of bound.
        """
        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            raise IndexError('Indices out of range')

        self.put(i, j, value)

    def __setitem__(self, tuple key, value):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))
        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[0]):
            # TODO: don't create temp object
            view = MakeLLSparseMatrixViewFromView(self, <PyObject *>key[0], <PyObject *>key[1])
            self.A.assign(view, value)

            del view
            return

        cdef INT_t i = key[0]
        cdef INT_t j = key[1]

        self.safe_put(i, j, <double> value)

    ####################################################################################################################
    #                                            *** GET ***
    cdef at(self, INT_t i, INT_t j):
        """
        Return element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        cdef INT_t k, t

        return self.A.safe_at(self.row_indices[i], self.col_indices[j])

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
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            return MakeLLSparseMatrixViewFromView(self, <PyObject *>key[0], <PyObject *>key[1])

        cdef INT_t i = key[0]
        cdef INT_t j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    #                                            *** COPY ***
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

        cdef SIZE_t size_hint
        cdef double val

        if compress:
            size_hint = self.count_nnz()
        else:
            size_hint = min(self.nrow * self.ncol, self.A.nalloc)

        cdef LLSparseMatrix A_copy = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, size_hint=size_hint, store_zeros=self.store_zeros)

        cdef:
            INT_t i
            INT_t row_index

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
    #                                            *** elements ***
    def get_matrix(self):
        """
        Return pointer to original matrix ``A``.
        """
        return self.A

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

    def count_nnz(self):
        if not self.__counted_nnz:
            self._nnz = self._count_nnz()
            self.__counted_nnz = True

        return self._nnz

    cdef INT_t _count_nnz(self):
        """
        Count number of non zeros elements.

        Returns:
            The number of non zeros elements if the corresponding :class:`LLSparseMatrix` doesn't store zeros, otherwise
            returns the ``size = nrow * ncol``.
        """
        self.assert_status_ok()

        cdef:
            INT_t i, j, row_index
            INT_t nnz = 0

        if self.store_zeros:
            nnz = self.nrow * self.ncol
        else:
            for i from 0 <= i < self.nrow:
                row_index = self.row_indices[i]
                for j from 0 <= j < self.ncol:
                    if self.A[row_index, self.col_indices[j]] != 0.0:
                        nnz += 1

        return nnz

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def __repr__(self):
        s = "LLSparseMatrixView of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
        return s



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
        Use only factory functions to create a view to a :class:`LLSparseMatrix`.

    """
    cdef:
        INT_t nrow
        INT_t * row_indices,
        INT_t ncol
        INT_t * col_indices
        INT_t A_nrow = A.nrow
        INT_t A_ncol = A.ncol

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


cdef LLSparseMatrixView MakeLLSparseMatrixViewFromView(LLSparseMatrixView A, PyObject* obj1, PyObject* obj2):
    """
    Factory function to create a new :class:`LLSparseMatrixView` for a :class:`LLSparseMatrixView`.

    Two index objects must be provided. Such objects can be:
        - an integer;
        - a list;
        - a slice;
        - a numpy array.

    Args:
        A: A :class:`LLSparseMatrixView` to be *viewed*.
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
        Use only factory functions to create a view to a :class:`LLSparseMatrixView`.

    """
    cdef:
        INT_t nrow
        INT_t * row_indices,
        INT_t ncol
        INT_t * col_indices
        INT_t A_nrow = A.nrow
        INT_t A_ncol = A.ncol
        INT_t i, j

    row_indices = create_c_array_indices_from_python_object(A_nrow, obj1, &nrow)
    col_indices = create_c_array_indices_from_python_object(A_ncol, obj2, &ncol)

    cdef LLSparseMatrixView view = LLSparseMatrixView(A.A, nrow, ncol)

    # construct arrays with adapted indices
    cdef INT_t * real_row_indices
    cdef INT_t * real_col_indices

    real_row_indices = <INT_t *> PyMem_Malloc(nrow * sizeof(INT_t))
    if not real_row_indices:
        raise MemoryError()

    real_col_indices = <INT_t *> PyMem_Malloc(ncol * sizeof(INT_t))
    if not real_col_indices:
        raise MemoryError()

    for i from 0 <= i < nrow:
        real_row_indices[i] = A.row_indices[row_indices[i]]

    for j from 0 <= j < ncol:
        real_col_indices[j] = A.col_indices[col_indices[j]]

    view.row_indices = real_row_indices
    view.col_indices = real_col_indices

    # free non used arrays
    PyMem_Free(row_indices)
    PyMem_Free(col_indices)

    if nrow == 0 or ncol == 0:
        view.is_empty = True
    else:
        view.is_empty = False

    view.__status_ok = True

    return view
