

# forward declaration
cdef class LLSparseMatrixView

from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject

cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef extern from "Python.h":
    # *** Types ***
    Py_ssize_t PY_SSIZE_T_MAX
    int PyInt_Check(PyObject *o)
    long PyInt_AS_LONG(PyObject *io)

    # *** Slices ***
    ctypedef struct PySliceObject:
        pass

    # Cython's version doesn't work for all versions...
    int PySlice_GetIndicesEx(
        PySliceObject* s, Py_ssize_t length,
        Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step,
        Py_ssize_t *slicelength) except -1

    int PySlice_Check(PyObject *ob)

    # *** List ***
    int PyList_Check(PyObject *p)
    PyObject* PyList_GetItem(PyObject *list, Py_ssize_t index)
    Py_ssize_t PyList_Size(PyObject *list)

cdef extern from "numpy/arrayobject.h":
    int PyArray_Check(PyObject * ob)
    ctypedef struct PyArrayObject:
        pass
    int PyArray_NDIM(PyArrayObject *arr)
    cnp.npy_intp PyArray_DIM(PyArrayObject* arr, int n)
    void *PyArray_DATA(PyArrayObject *arr)

    # TODO: check type of elements inside an numpy array...
    ctypedef struct PyArray_Descr:
        char type
        int type_num
        char kind
    PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)


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

    def copy(self, compress=True):
        """
        Create a new :class:`LLSparseMatrix` from the view and return it.

        Args:
            compress: If ``True``, we use the minimum size for the matrix.

        """
        cdef int size_hint
        cdef double val

        if compress:
            size_hint = self.count_nnz()
        else:
            size_hint = min(self.nrow * self.ncol, self.A.nalloc)

        cdef LLSparseMatrix A_copy = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, size_hint=size_hint)

        cdef:
            int i
            int row_index

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
                    if val != 0.0:
                        A_copy[i, j] = self.A[row_index, self.col_indices[j]]

        return A_copy

    cdef int count_nnz(self):
        """
        Count number of non zeros elements.

        Returns:
            The number of non zeros elements if the corresponding :class:`LLSparseMatrix` doesn't store zeros, otherwise
            returns the ``size = nrow * ncol``.
        """
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
        This sould be the only way to create a view to a :class:`LLSparseMatrix`

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


cdef int * create_c_array_indices_from_python_object(int max_length, PyObject * obj, int * number_of_elements) except NULL:
    """

    Args:
        max_length: Bound on the indices that can not be crossed. Indices must be **strictly** smaller than ``max_length``.
            Consider this as ``A.nrow`` or ``A.ncol``.
        obj: A Python object. Should be an integer, a slice, a list or a numpy array.
        number_of_elements: Number of elements returned in the C-array of indices. This is an OUT argument.

    Returns:
        A C-array with the corresponding indices of length ``number_of_elements``.

    Warning:
        There are not many tests about the indices given in a list or a numpy array.

        We do test:
            * if indices are out of bound, i.e. 0 <= index < max_length

        We do not test:
            * their relative order;
            * their uniqueness.

        We partially test:
            * if elements inside the index objects are integers: this is done for list but **not** for numpy arrays...
    """
    cdef int ret
    cdef Py_ssize_t start, stop, step, length, index

    cdef int i, j
    cdef int * indices
    cdef PyObject *val

    cdef int array_dim
    cdef int * array_data
    cdef PyArray_Descr * py_array_descr

    # CASES
    # Integer
    if PyInt_Check(obj):
        i = <int> PyInt_AS_LONG(obj)
        length = 1
        indices = <int *> PyMem_Malloc(length * sizeof(int))
        if not indices:
            raise MemoryError()

        indices[0] = i

    # Slice
    elif PySlice_Check(obj):
        # slice
        ret = PySlice_GetIndicesEx(<PySliceObject*>obj, max_length, &start, &stop, &step, &length)
        if ret:
            raise RuntimeError("Slice could not be translated")

        #print "start, stop, step, length = (%d, %d, %d, %d)" % (start, stop, step, length)

        indices = <int *> PyMem_Malloc(length * sizeof(int))
        if not indices:
            raise MemoryError()

        # populate indices
        i = start
        for j from 0 <= j < length:
            indices[j] = i
            i += step

    # List
    elif PyList_Check(obj):
        length = PyList_Size(obj)
        indices = <int *> PyMem_Malloc(length * sizeof(int))
        if not indices:
            raise MemoryError()

        for i from 0 <= i < length:
            val = PyList_GetItem(obj, <Py_ssize_t>i)
            if PyInt_Check(val):
                index = PyInt_AS_LONG(val)
                # test if index is valid
                if not (0 <= index < max_length):
                    raise IndexError("Index %d out of bounds [%d, %d[" % (<long>index, 0, max_length))
                indices[i] = <int> index
            else:
                PyMem_Free(indices)
                raise ValueError("List must only contain integers")

    # numpy array
    elif PyArray_Check(obj):
        array_dim = <int> PyArray_NDIM(<PyArrayObject *>obj)
        if array_dim != 1:
            raise IndexError("Numpy array must be of dimension 1")

        length = <Py_ssize_t> PyArray_DIM(<PyArrayObject *>obj, 0)

        indices = <int *> PyMem_Malloc(length * sizeof(int))
        if not indices:
            raise MemoryError()

        # TODO: remove or control this is not DANGEROUS
        array_data = <int* > PyArray_DATA(<PyArrayObject *> obj)

        # test type of array elements
        # TODO: I don't know how to find out what the type of elements is!!!!
        #py_array_descr = PyArray_DTYPE(<PyArrayObject*> obj)

        # we cannot copy the C-array directly as we must test each element
        for i from 0 <= i < length:
            index = <int> array_data[i]
            if not (0 <= index < max_length):
                raise IndexError("Index %d out of bounds [%d, %d[" % (<long>index, 0, max_length))
            indices[i] = array_data[i]

    else:
        raise TypeError("Index object is not recognized to create a LLSparseMatrixView")

    number_of_elements[0] = <int> length

    return indices