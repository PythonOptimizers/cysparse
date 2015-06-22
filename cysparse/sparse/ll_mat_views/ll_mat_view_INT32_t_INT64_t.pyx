"""
Lightweight object to view a :class:`LLSparseMatrix_INT32_t_INT64_t`.


"""
from cysparse.types.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView_INT32_t_INT64_t

from cysparse.sparse.s_mat cimport unexposed_value
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT64_t cimport LLSparseMatrix_INT32_t_INT64_t
from cysparse.sparse.sparse_utils.generic.generate_indices_INT32_t cimport create_c_array_indices_from_python_object_INT32_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject
from python_ref cimport Py_INCREF, Py_DECREF

cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cdef class LLSparseMatrixView_INT32_t_INT64_t:
    def __cinit__(self,control_object, LLSparseMatrix_INT32_t_INT64_t A, INT32_t nrow, INT32_t ncol):
        assert control_object == unexposed_value, "LLSparseMatrixView must be instantiated with a factory method"
        self.nrow = nrow  # number of rows of the view
        self.ncol = ncol  # number of columns of the view

        self.type = "LLSparseMatrixView"
        self.__type_name = "LLSparseMatrixView [INT32_t, INT64_t]"

        self.is_empty = True

        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.__is_symmetric = A.is_symmetric
        self.__store_zeros = A.store_zeros

        self.__counted_nnz = False
        self._nnz = 0


    def __dealloc__(self):
        PyMem_Free(self.row_indices)
        PyMem_Free(self.col_indices)

        Py_DECREF(self.A) # release ref

    ####################################################################################################################
    # Set/Get individual elements
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    cdef put(self, INT32_t i, INT32_t j, INT64_t value):
        self.A.put(self.row_indices[i], self.col_indices[j], value)

    cdef int safe_put(self, INT32_t i, INT32_t j, INT64_t value) except -1:
        """
        Set ``A_view[i, j] = value`` directly.

        Raises:
            IndexError: when index out of bound.
        """
        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            raise IndexError('Indices out of range')
            return -1

        self.put(i, j, value)
        return 1

    ####################################################################################################################
    #                                            *** GET ***
    cdef INT64_t at(self, INT32_t i, INT32_t j):
        """
        Return element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        return self.A.safe_at(self.row_indices[i], self.col_indices[j])

    # EXPLICIT TYPE TESTS

    cdef INT64_t safe_at(self, INT32_t i, INT32_t j) except? 1:

        """
        Return element ``(i, j)`` but with check for out of bounds indices.

        Raises:
            IndexError: when index out of bound.

        """
        if not 0 <= i < self.nrow or not 0 <= j < self.ncol:
            raise IndexError("Index out of bounds")

            return 1


        return self.at(i, j)

    ####################################################################################################################
    # __setitem/__getitem__
    ####################################################################################################################
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            return MakeLLSparseMatrixViewFromView_INT32_t_INT64_t(self, <PyObject *>key[0], <PyObject *>key[1])

        cdef INT32_t i = key[0]
        cdef INT32_t j = key[1]

        return self.safe_at(i, j)

    def __setitem__(self, tuple key, value):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))
        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[0]):
            # TODO: don't create temp object
            view = MakeLLSparseMatrixViewFromView_INT32_t_INT64_t(self, <PyObject *>key[0], <PyObject *>key[1])
            self.A.assign(view, value)

            del view
            return

        cdef INT32_t i = key[0]
        cdef INT32_t j = key[1]

        self.safe_put(i, j, <INT64_t> value)

    ####################################################################################################################
    # COPY
    ####################################################################################################################
    def matrix_copy(self, compress=True):
        """
        Create a new :class:`LLSparseMatrix_INT32_t_INT64_t` from the view and return it.

        Args:
            compress: If ``True``, we use the minimum size for the matrix.

        Note:
            Because we lost sight of zero elements added in the viewed :class:`LLSparseMatrix_INT32_t_INT64_t`,
            the returned matrix has its ``store_zeros`` attribute set
            to ``False`` and no zero is copied.

            Because we don't know what submatrix is taken, the returned matrix **cannot** by symmetric.

        """
        # This is completely arbitrary
        cdef INT32_t size_hint = min(<INT32_t>(self.nrow * self.ncol)/4, self.A.nalloc) + 1

        cdef LLSparseMatrix_INT32_t_INT64_t A_copy = LLSparseMatrix_INT32_t_INT64_t(control_object=unexposed_value,
                                                                                  nrow=self.nrow,
                                                                                  ncol=self.ncol,
                                                                                  size_hint=size_hint,
                                                                                  store_zeros=False,
                                                                                  __is_symmetric=False)

        cdef:
            INT32_t i, j
            INT32_t row_index

        for i from 0 <= i < self.nrow:
            row_index = self.row_indices[i]
            for j from 0 <= j < self.ncol:
                A_copy.put(i, j, self.A[row_index, self.col_indices[j]])

        if compress:
            A_copy.compress()

        return A_copy

    def copy(self):
        """
        Create a new :class:`LLSparseMatrixView_INT32_t_INT64_t` from this object.

        """
        cdef:
            INT32_t nrow
            INT32_t * row_indices,
            INT32_t ncol
            INT32_t * col_indices

        row_indices = <INT32_t *> PyMem_Malloc(self.nrow * sizeof(INT32_t))
        if not row_indices:
            raise MemoryError()

        col_indices = <INT32_t *> PyMem_Malloc(self.ncol * sizeof(INT32_t))
        if not col_indices:
            PyMem_Free(row_indices)
            raise MemoryError()

        cdef LLSparseMatrixView_INT32_t_INT64_t view = LLSparseMatrixView_INT32_t_INT64_t(unexposed_value, self.A, self.nrow, self.ncol)

        for i from 0 <= i < self.nrow:
            row_indices[i] = self.row_indices[i]

        for j from 0 <= j < self.ncol:
            col_indices[j] = self.col_indices[j]

        view.row_indices = row_indices
        view.col_indices = col_indices

        view.is_empty = self.is_empty
        view.__counted_nnz = self.__counted_nnz
        view._nnz = self._nnz

        return view

    ####################################################################################################################
    # OUTPUT STRINGS
    ####################################################################################################################
    def attributes_short_string(self):
        """

        """
        s = "of dim %d by %d" % (self.nrow, self.ncol)
        return s

    def attributes_long_string(self):

        symmetric_string = None
        if self.__is_symmetric:
            symmetric_string = 'symmetric'
        else:
            symmetric_string = 'general'

        store_zeros_string = None
        if self.__store_zeros:
            store_zeros_string = "store_zeros"
        else:
            store_zeros_string = "no_zeros"

        s = "%s [%s, %s]" % (self.attributes_short_string(), symmetric_string, store_zeros_string)

        return s

    def attributes_condensed(self):
        symmetric_string = None
        if self.__is_symmetric:
            symmetric_string = 'S'
        else:
            symmetric_string = 'G'

        store_zeros_string = None
        if self.__store_zeros:
            store_zeros_string = "SZ"
        else:
            store_zeros_string = "NZ"

        s= "(%s, %s, [%d, %d])" % (symmetric_string, store_zeros_string, self.nrow, self.ncol)

        return s

    def _matrix_description_before_printing(self):
        s = "%s %s" % (self.__type_name, self.attributes_condensed())
        return s

    def __repr__(self):
        s = "%s %s" % (self.__type_name, self.attributes_long_string())
        return s

########################################################################################################################
# Factory methods
########################################################################################################################
cdef LLSparseMatrixView_INT32_t_INT64_t MakeLLSparseMatrixView_INT32_t_INT64_t(LLSparseMatrix_INT32_t_INT64_t A, PyObject* obj1, PyObject* obj2):
    """
    Factory function to create a new :class:`LLSparseMatrixView_INT32_t_INT64_t` for a :class:`LLSparseMatrix_INT32_t_INT64_t`.

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
        A corresponding :class:`LLSparseMatrixView_INT32_t_INT64_t`. This view can be empty with the wrong index objects.

    Warning:
        Use only factory functions to create a view to a :class:`LLSparseMatrix_INT32_t_INT64_t`.

    """
    cdef:
        INT32_t nrow
        INT32_t * row_indices,
        INT32_t ncol
        INT32_t * col_indices
        INT32_t A_nrow = A.__nrow
        INT32_t A_ncol = A.__ncol

    row_indices = create_c_array_indices_from_python_object_INT32_t(A_nrow, obj1, &nrow)
    col_indices = create_c_array_indices_from_python_object_INT32_t(A_ncol, obj2, &ncol)

    cdef LLSparseMatrixView_INT32_t_INT64_t view = LLSparseMatrixView_INT32_t_INT64_t(unexposed_value, A, nrow, ncol)

    view.row_indices = row_indices
    view.col_indices = col_indices

    if nrow == 0 or ncol == 0:
        view.is_empty = True
    else:
        view.is_empty = False

    return view


cdef LLSparseMatrixView_INT32_t_INT64_t MakeLLSparseMatrixViewFromView_INT32_t_INT64_t(LLSparseMatrixView_INT32_t_INT64_t A, PyObject* obj1, PyObject* obj2):
    """
    Factory function to create a new :class:`LLSparseMatrixView_INT32_t_INT64_t` for a :class:`LLSparseMatrixView_INT32_t_INT64_t`.

    Two index objects must be provided. Such objects can be:
        - an integer;
        - a list;
        - a slice;
        - a numpy array.

    Args:
        A: A :class:`LLSparseMatrixView_INT32_t_INT64_t` to be *viewed*.
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
        A corresponding :class:`LLSparseMatrixView_INT32_t_INT64_t`. This view can be empty with the wrong index objects.

    Warning:
        Use only factory functions to create a view to a :class:`LLSparseMatrixView_INT32_t_INT64_t`.

    """
    cdef:
        INT32_t nrow
        INT32_t * row_indices,
        INT32_t ncol
        INT32_t * col_indices
        INT32_t A_nrow = A.__nrow
        INT32_t A_ncol = A.__ncol
        INT32_t i, j

    row_indices = create_c_array_indices_from_python_object_INT32_t(A_nrow, obj1, &nrow)
    col_indices = create_c_array_indices_from_python_object_INT32_t(A_ncol, obj2, &ncol)

    cdef LLSparseMatrixView_INT32_t_INT64_t view = LLSparseMatrixView_INT32_t_INT64_t(unexposed_value, A.A, nrow, ncol)

    # construct arrays with adapted indices
    cdef INT32_t * real_row_indices
    cdef INT32_t * real_col_indices

    real_row_indices = <INT32_t *> PyMem_Malloc(nrow * sizeof(INT32_t))
    if not real_row_indices:
        raise MemoryError()

    real_col_indices = <INT32_t *> PyMem_Malloc(ncol * sizeof(INT32_t))
    if not real_col_indices:
        PyMem_Free(real_row_indices)
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

    return view