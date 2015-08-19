"""
Lightweight object to view a :class:`LLSparseMatrix_INT64_t_INT64_t`.


"""
from cysparse.types.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView_INT64_t_INT64_t

from cysparse.sparse.s_mat cimport unexposed_value
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT64_t cimport LLSparseMatrix_INT64_t_INT64_t
from cysparse.sparse.sparse_utils.generic.generate_indices_INT64_t cimport create_c_array_indices_from_python_object_INT64_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject
from cpython cimport Py_INCREF, Py_DECREF

cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cdef class LLSparseMatrixView_INT64_t_INT64_t:
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    def __cinit__(self,control_object, LLSparseMatrix_INT64_t_INT64_t A, INT64_t nrow, INT64_t ncol):
        assert control_object == unexposed_value, "LLSparseMatrixView must be instantiated with a factory method"
        self.__nrow = nrow  # number of rows of the view
        self.__ncol = ncol  # number of columns of the view

        self.__type = "LLSparseMatrixView"
        self.__type_name = "LLSparseMatrixView [INT64_t, INT64_t]"

        self.__is_empty = True

        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly


    
    @property
    def nrow(self):
        return self.__nrow

    
    @property
    def ncol(self):
        return self.__ncol

    
    @property
    def nnz(self):
        """
        Return the number of non zeros inside the view.

        Warning:
            This property is costly.
        """
        # Because views must **always** be up to date with the original matrix, we cannot rely on cached results.
        # We could use a cache in LLSparseMatrix but is it worth it?
        return self.A.count_nnz_from_indices(self.row_indices, self.__nrow, self.col_indices, self.__ncol, count_only_stored=True)

    # for compatibility with numpy, PyKrylov, etc
    
    @property
    def shape(self):
        return (self.__nrow, self.__ncol)

    
    @property
    def dtype(self):
        return self.A.cp_type.dtype

    
    @property
    def itype(self):
        return self.A.cp_type.itype

    
    @property
    def is_symmetric(self):
        return self.A.__is_symmetric

    
    @property
    def is_mutable(self):
        return self.A.__is_mutable

    
    @property
    def store_zeros(self):
        return self.A.__store_zeros

    
    @property
    def is_empty(self):
        return self.__is_empty

    
    @property
    def type(self):
        return self.__type

    
    @property
    def type_name(self):
        return self.__type_name

    def get_matrix(self):
        """
        Return pointer to original matrix ``A``.
        """
        return self.A

    def __dealloc__(self):
        PyMem_Free(self.row_indices)
        PyMem_Free(self.col_indices)

        Py_DECREF(self.A) # release ref

    ####################################################################################################################
    # Set/Get individual elements
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    cdef put(self, INT64_t i, INT64_t j, INT64_t value):
        self.A.put(self.row_indices[i], self.col_indices[j], value)

    cdef int safe_put(self, INT64_t i, INT64_t j, INT64_t value) except -1:
        """
        Set ``A_view[i, j] = value`` directly.

        Raises:
            IndexError: when index out of bound.
        """
        if i < 0 or i >= self.__nrow or j < 0 or j >= self.__ncol:
            raise IndexError('Indices out of range')

        self.put(i, j, value)
        return 1

    ####################################################################################################################
    #                                            *** GET ***
    cdef INT64_t at(self, INT64_t i, INT64_t j):
        """
        Return element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        return self.A.safe_at(self.row_indices[i], self.col_indices[j])

    # EXPLICIT TYPE TESTS

    cdef INT64_t safe_at(self, INT64_t i, INT64_t j) except? 1:

        """
        Return element ``(i, j)`` but with check for out of bounds indices.

        Raises:
            IndexError: when index out of bound.

        """
        if not 0 <= i < self.__nrow or not 0 <= j < self.__ncol:
            raise IndexError("Index out of bounds")


        return self.at(i, j)

    ####################################################################################################################
    # __setitem/__getitem__
    ####################################################################################################################
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            return MakeLLSparseMatrixViewFromView_INT64_t_INT64_t(self, <PyObject *>key[0], <PyObject *>key[1])

        cdef INT64_t i = key[0]
        cdef INT64_t j = key[1]

        return self.safe_at(i, j)

    def __setitem__(self, tuple key, value):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))
        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[0]):
            # TODO: don't create temp object
            view = MakeLLSparseMatrixViewFromView_INT64_t_INT64_t(self, <PyObject *>key[0], <PyObject *>key[1])
            self.A.assign(view, value)

            del view
            return

        cdef INT64_t i = key[0]
        cdef INT64_t j = key[1]

        self.safe_put(i, j, <INT64_t> value)

    ####################################################################################################################
    # COPY
    ####################################################################################################################
    def matrix_copy(self, compress=True):
        """
        Create a new :class:`LLSparseMatrix_INT64_t_INT64_t` from the view and return it.

        Args:
            compress: If ``True``, we use the minimum size for the matrix.

        Note:
            Because we lost sight of zero elements added in the viewed :class:`LLSparseMatrix_INT64_t_INT64_t`,
            the returned matrix has its ``store_zeros`` attribute set
            to ``False`` and no zero is copied.

            Because we don't know what submatrix is taken, the returned matrix **cannot** by symmetric.

        """
        # This is completely arbitrary
        cdef INT64_t size_hint = min(<INT64_t>(self.__nrow * self.__ncol)/4, self.A.nalloc) + 1

        cdef LLSparseMatrix_INT64_t_INT64_t A_copy = LLSparseMatrix_INT64_t_INT64_t(control_object=unexposed_value,
                                                                                  nrow=self.__nrow,
                                                                                  ncol=self.__ncol,
                                                                                  size_hint=size_hint,
                                                                                  store_zeros=False,
                                                                                  is_symmetric=False)

        cdef:
            INT64_t i, j
            INT64_t row_index

        for i from 0 <= i < self.__nrow:
            row_index = self.row_indices[i]
            for j from 0 <= j < self.__ncol:
                A_copy.put(i, j, self.A[row_index, self.col_indices[j]])

        if compress:
            A_copy.compress()

        return A_copy

    def copy(self):
        """
        Create a new :class:`LLSparseMatrixView_INT64_t_INT64_t` from this object.

        """
        cdef:
            INT64_t * row_indices,
            INT64_t * col_indices

        row_indices = <INT64_t *> PyMem_Malloc(self.__nrow * sizeof(INT64_t))
        if not row_indices:
            raise MemoryError()

        col_indices = <INT64_t *> PyMem_Malloc(self.__ncol * sizeof(INT64_t))
        if not col_indices:
            PyMem_Free(row_indices)
            raise MemoryError()

        cdef LLSparseMatrixView_INT64_t_INT64_t view = LLSparseMatrixView_INT64_t_INT64_t(unexposed_value, self.A, self.__nrow, self.__ncol)

        for i from 0 <= i < self.__nrow:
            row_indices[i] = self.row_indices[i]

        for j from 0 <= j < self.__ncol:
            col_indices[j] = self.col_indices[j]

        view.row_indices = row_indices
        view.col_indices = col_indices

        view.__is_empty = self.__is_empty

        return view

    ####################################################################################################################
    # OUTPUT STRINGS
    ####################################################################################################################
    def attributes_short_string(self):
        """

        """
        return self.A.attributes_short_string()

    def attributes_long_string(self):
        return self.A.attributes_long_string()

    def attributes_condensed(self):
        return self.A.attributes_condensed()

    def _matrix_description_before_printing(self):
        return self._matrix_description_before_printing()

    def __repr__(self):
        return 'View to ' + self.A.__repr__()

    def __str__(self):
        """
        Return a string representing the view of the matrix.

        Warning:
            This method is costly! Use with care.
        """
        return 'View to ' + str(self.matrix_copy())

########################################################################################################################
# Factory methods
########################################################################################################################
cdef LLSparseMatrixView_INT64_t_INT64_t MakeLLSparseMatrixView_INT64_t_INT64_t(LLSparseMatrix_INT64_t_INT64_t A, PyObject* obj1, PyObject* obj2):
    """
    Factory function to create a new :class:`LLSparseMatrixView_INT64_t_INT64_t` for a :class:`LLSparseMatrix_INT64_t_INT64_t`.

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
        A corresponding :class:`LLSparseMatrixView_INT64_t_INT64_t`. This view can be empty with the wrong index objects.

    Warning:
        Use only factory functions to create a view to a :class:`LLSparseMatrix_INT64_t_INT64_t`.

    """
    cdef:
        INT64_t nrow
        INT64_t * row_indices,
        INT64_t ncol
        INT64_t * col_indices
        INT64_t A_nrow = A.__nrow
        INT64_t A_ncol = A.__ncol

    row_indices = create_c_array_indices_from_python_object_INT64_t(A_nrow, obj1, &nrow)
    col_indices = create_c_array_indices_from_python_object_INT64_t(A_ncol, obj2, &ncol)

    cdef LLSparseMatrixView_INT64_t_INT64_t view = LLSparseMatrixView_INT64_t_INT64_t(unexposed_value, A, nrow, ncol)

    view.row_indices = row_indices
    view.col_indices = col_indices

    if nrow == 0 or ncol == 0:
        view.__is_empty = True
    else:
        view.__is_empty = False

    return view


cdef LLSparseMatrixView_INT64_t_INT64_t MakeLLSparseMatrixViewFromView_INT64_t_INT64_t(LLSparseMatrixView_INT64_t_INT64_t A, PyObject* obj1, PyObject* obj2):
    """
    Factory function to create a new :class:`LLSparseMatrixView_INT64_t_INT64_t` for a :class:`LLSparseMatrixView_INT64_t_INT64_t`.

    Two index objects must be provided. Such objects can be:
        - an integer;
        - a list;
        - a slice;
        - a numpy array.

    Args:
        A: A :class:`LLSparseMatrixView_INT64_t_INT64_t` to be *viewed*.
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
        A corresponding :class:`LLSparseMatrixView_INT64_t_INT64_t`. This view can be empty with the wrong index objects.

    Warning:
        Use only factory functions to create a view to a :class:`LLSparseMatrixView_INT64_t_INT64_t`.

    """
    cdef:
        INT64_t nrow
        INT64_t * row_indices
        INT64_t ncol
        INT64_t * col_indices
        INT64_t A_nrow = A.__nrow
        INT64_t A_ncol = A.__ncol
        INT64_t i, j

    row_indices = create_c_array_indices_from_python_object_INT64_t(A_nrow, obj1, &nrow)
    col_indices = create_c_array_indices_from_python_object_INT64_t(A_ncol, obj2, &ncol)

    cdef LLSparseMatrixView_INT64_t_INT64_t view = LLSparseMatrixView_INT64_t_INT64_t(unexposed_value, A.A, nrow, ncol)

    # construct arrays with adapted indices
    cdef INT64_t * real_row_indices
    cdef INT64_t * real_col_indices

    real_row_indices = <INT64_t *> PyMem_Malloc(nrow * sizeof(INT64_t))
    if not real_row_indices:
        raise MemoryError()

    real_col_indices = <INT64_t *> PyMem_Malloc(ncol * sizeof(INT64_t))
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
        view.__is_empty = True
    else:
        view.__is_empty = False

    return view