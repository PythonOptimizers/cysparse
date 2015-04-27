"""
ll_mat extension.

{{COMPLEX: NO}}
{{GENERIC TYPES: NO}}
"""
from __future__ import print_function

from sparse_lib.cysparse_types cimport *

from sparse_lib.sparse.sparse_mat cimport MutableSparseMatrix
from sparse_lib.sparse.csr_mat cimport MakeCSRSparseMatrix, MakeCSRComplexSparseMatrix
from sparse_lib.sparse.csc_mat cimport MakeCSCSparseMatrix
#from sparse_lib.utils.equality cimport values_are_equal
from sparse_lib.sparse.IO.mm cimport MakeLLSparseMatrixFromMMFile2, MakeMMFileFromSparseMatrix


# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

cnp.import_array()

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc,free, calloc
from libc.string cimport memcpy
from cpython cimport PyObject, Py_INCREF

# TODO: use more internal CPython code
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)
    int PyComplex_Check(PyObject * o)
    double PyComplex_RealAsDouble(PyObject *op)
    double PyComplex_ImagAsDouble(PyObject *op)
    PyObject* PyComplex_FromDoubles(double real, double imag)


cdef INT_t LL_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default
cdef FLOAT_t LL_MAT_INCREASE_FACTOR = 1.5      # reallocating factor if size is not enough, must be > 1
cdef INT_t LL_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef INT_t LL_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format
# same for complex matrix printing
cdef INT_t LL_MAT_COMPLEX_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef INT_t LL_MAT_COMPLEX_PPRINT_COL_THRESH = 10        # column threshold for choosing print format


cdef extern from "Python.h":
    PyObject* Py_BuildValue(char *format, ...)
    PyObject* PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(PyObject *list, Py_ssize_t i, PyObject *o)
    PyObject* PyFloat_FromDouble(double v)

# forward declaration
cdef class LLSparseMatrix(MutableSparseMatrix)

from sparse_lib.sparse.ll_mat_view cimport LLSparseMatrixView, MakeLLSparseMatrixView

cdef class LLSparseMatrix(MutableSparseMatrix):
    """
    Linked-List Format matrix.

    Note:
        Despite its name, this matrix doesn't use any linked list.
    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    def __cinit__(self, INT_t nrow, INT_t ncol, SIZE_t size_hint=LL_MAT_DEFAULT_SIZE_HINT,
                  bint is_symmetric=False, bint store_zeros=False, is_complex=False):
        """
        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if size_hint < 1:
            raise ValueError('size_hint (%d) must be >= 1' % size_hint)

        self.type_name = "LLSparseMatrix"

        val = <FLOAT_t *> PyMem_Malloc(self.size_hint * sizeof(FLOAT_t))
        if not val:
            raise MemoryError()
        self.val = val

        if self.is_complex:
            ival = <FLOAT_t *> PyMem_Malloc(self.size_hint * sizeof(FLOAT_t))
            if not ival:
                raise MemoryError()
            self.ival = ival

        col = <INT_t *> PyMem_Malloc(self.size_hint * sizeof(INT_t))
        if not col:
            raise MemoryError()
        self.col = col

        link = <INT_t *> PyMem_Malloc(self.size_hint * sizeof(INT_t))
        if not link:
            raise MemoryError()
        self.link = link

        root = <INT_t *> PyMem_Malloc(self.nrow * sizeof(INT_t))
        if not root:
            raise MemoryError()
        self.root = root

        self.nalloc = self.size_hint
        self.free = -1

        cdef INT_t i
        for i from 0 <= i < nrow:
            root[i] = -1

    def __dealloc__(self):
        """
        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        PyMem_Free(self.val)
        if self.is_complex:
            PyMem_Free(self.ival)
        PyMem_Free(self.col)
        PyMem_Free(self.link)
        PyMem_Free(self.root)

    cdef _realloc(self, INT_t nalloc_new):
        """
        Realloc space for the internal arrays.

        Note:
            Internal arrays can be expanded or shrunk.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef:
            void *temp

        temp = <INT_t *> PyMem_Realloc(self.col, nalloc_new * sizeof(INT_t))
        if not temp:
            raise MemoryError()
        self.col = <INT_t*>temp

        temp = <INT_t *> PyMem_Realloc(self.link, nalloc_new * sizeof(INT_t))
        if not temp:
            raise MemoryError()
        self.link = <INT_t *>temp

        temp = <FLOAT_t *> PyMem_Realloc(self.val, nalloc_new * sizeof(FLOAT_t))
        if not temp:
            raise MemoryError()
        self.val = <FLOAT_t *>temp

        if self.is_complex:
            temp = <FLOAT_t *> PyMem_Realloc(self.ival, nalloc_new * sizeof(FLOAT_t))
            if not temp:
                raise MemoryError()
            self.ival = <FLOAT_t *>temp

        self.nalloc = nalloc_new

    cdef _realloc_expand(self):
        """
        Realloc space for internal arrays.

        Note:
            We use ``LL_MAT_INCREASE_FACTOR`` as expanding factor.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        assert LL_MAT_INCREASE_FACTOR > 1.0
        cdef INT_t real_new_alloc = <INT_t>(<FLOAT_t>LL_MAT_INCREASE_FACTOR * self.nalloc) + 1

        return self._realloc(real_new_alloc)

    def compress(self):
        """
        Shrink matrix to its minimal size.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef:
            INT_t i, k, k_next, k_last, k_new, nalloc_new;

        nalloc_new = self.nnz  # new size for val, col and link arrays

        # remove entries with k >= nalloc_new from free list
        k_last = -1
        k = self.free
        while k != -1:
            k_next =  self.link[k]
            if k >= nalloc_new:
                if k_last == -1:
                    self.free = k_next
                else:
                    self.link[k_last] = k_next
            else:
                k_last = k
                k = k_next

        # reposition matrix entries with k >= nalloc_new
        for i from 0 <= i < self.nrow:
            k_last = -1
            k = self.root[i]
            while k != -1:
                if k >= nalloc_new:
                    k_new = self.free
                    if k_last == -1:
                        self.root[i] = k_new
                    else:
                        self.link[k_last] = k_new
                    self.free = self.link[k_new]
                    self.val[k_new] = self.val[k]
                    if self.is_complex:
                        self.ival[k] = self.ival[k]
                    self.col[k_new] = self.col[k]
                    self.link[k_new] = self.link[k]
                    k_last = k_new
                else:
                    k_last = k

                k = self.link[k]

        # shrink arrays
        self._realloc(nalloc_new)

        return

    def memory_real(self):
        """
        Return the real amount of memory used internally for the matrix.

        Returns:
            The exact number of bits used to store the matrix (but not the object in itself, only the internal memory
            needed to store the matrix).

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef INT_t total_memory = 0

        # root
        total_memory += self.nrow * sizeof(INT_t)
        # col
        total_memory += self.nalloc * sizeof(INT_t)
        # link
        total_memory += self.nalloc * sizeof(INT_t)
        # val
        total_memory += self.nalloc * sizeof(FLOAT_t)

        if self.is_complex:
            # ival
            total_memory += self.nalloc * sizeof(FLOAT_t)

        return total_memory

    ####################################################################################################################
    # Set/Get items
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    cdef put(self, INT_t i, INT_t j, FLOAT_t value, FLOAT_t imaginary = 0.0):
        """
        Set :math:`A[i, j] = \textrm{value}` or :math:`A[i, j] = \textrm{complex}` directly.

        Note:
            Store zero elements **only** if ``store_zeros`` is ``True``.

        Warning:
            No out of bound check.

        See:
            :meth:`safe_put`.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if self.is_symmetric and i < j:
            raise IndexError('Write operation to upper triangle of symmetric matrix not allowed')

        cdef INT_t k, new_elem, last, col

        # Find element to be set (or removed)
        col = last = -1
        k = self.root[i]
        while k != -1:
            col = self.col[k]
            # TODO: check this
            if col >= j:
                break
            last = k
            k = self.link[k]

        # Store value
        if self.store_zeros or (value != 0.0 or imaginary != 0.0):
            if col == j:
                # element already exist
                self.val[k] = value
                if self.is_complex:
                    self.ival[k] = imaginary
            else:
                # new element
                # find location for new element
                if self.free != -1:
                    # use element from the free chain
                    new_elem = self.free
                    self.free = self.link[new_elem]

                else:
                    # append new element to the end
                    new_elem = self.nnz

                # test if there is space for a new element
                if self.nnz == self.nalloc:
                    # we have to reallocate some space
                    self._realloc_expand()

                self.val[new_elem] = value
                if self.is_complex:
                    self.ival[new_elem] = imaginary

                self.col[new_elem] = j
                self.link[new_elem] = k

                if last == -1:
                    self.root[i] = new_elem
                else:
                    self.link[last] = new_elem

                self.nnz += 1

        else:
            # value == 0.0 and imaginary == 0.0:
            # if element exists but we don't store zero elements
            # we need to "zeroify" this element
            if col == j:
                # relink row i
                if last == -1:
                    self.root[i] = self.link[k]
                else:
                    self.link[last] = self.link[k]

                # add element to free list
                self.link[k] = self.free
                self.free = k

                self.nnz -= 1


    cdef safe_put(self, INT_t i, INT_t j, FLOAT_t value, FLOAT_t imaginary = 0.0):
        """
        Set ``A[i, j] = value`` directly.

        Raises:
            IndexError: when index out of bound.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            raise IndexError('Indices out of range')

        self.put(i, j, value, imaginary)

    cdef assign(self, LLSparseMatrixView view, object obj):
        """
        Set ``A[..., ...] = value`` directly.


        {{COMPLEX: NO}}
        {{GENERIC TYPES: YES}}
        """
        # test if view correspond...
        assert self == view.A

        if self.is_complex:
            raise NotImplemented("This operation is not (yet) implemented for complex matrices")

        update_ll_mat_matrix_from_c_arrays_indices_assign(self, view.row_indices, view.nrow,
                                                       view.col_indices, view.ncol, obj)

    def put_triplet(self,  list val, list index_i, list index_j):
        """
        Assign triplet :math:`\{(i, j, \textrm{val})\}` values to the matrix..


        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef Py_ssize_t index_i_length = len(index_i)
        cdef Py_ssize_t index_j_length = len(index_j)
        cdef Py_ssize_t val_length = len(val)

        assert index_j_length == index_j_length == val_length, "All lists must be of equal length"

        cdef Py_ssize_t i
        cdef PyObject * elem

        if self.is_complex:
            for i from 0 <= i < index_i_length:
                elem = <PyObject *> val[i]
                assert PyComplex_Check(elem), "Complex matrix takes only complex elements"
                self.safe_put(index_i[i], index_j[i], <FLOAT_t>PyComplex_RealAsDouble(elem), <FLOAT_t>PyComplex_ImagAsDouble(elem))
        else:
            for i from 0 <= i < index_i_length:
                self.safe_put(index_i[i], index_j[i], val[i])

    def __setitem__(self, tuple key, value):
        """
        A[i, j] = value

        Raises:
            IndexError: when index out of bound.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        cdef LLSparseMatrixView view

        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            # TODO: don't create temp object
            view = MakeLLSparseMatrixView(self, <PyObject *>key[0], <PyObject *>key[1])
            self.assign(view, value)

            del view
            return

        cdef INT_t i = key[0]
        cdef INT_t j = key[1]

        if self.is_complex:
            elem = <PyObject *> value
            assert PyComplex_Check(elem), "Complex matrix takes only complex ellements"
            self.safe_put(i, j, <FLOAT_t>PyComplex_RealAsDouble(elem), <FLOAT_t>PyComplex_ImagAsDouble(elem))

        else:
            self.safe_put(i, j, <FLOAT_t> value)

    ####################################################################################################################
    #                                            *** GET ***
    cdef at(self, INT_t i, INT_t j):
        """
        Return element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef INT_t k, t

        if self.is_symmetric and i < j:
            t = i; i = j; j = t

        k = self.root[i]

        while k != -1:
            # TODO: check this: we go over all elements in row i: is it really necessary?
            if self.col[k] == j:
                if self.is_complex:
                    return <object>PyComplex_FromDoubles(self.val[k], self.ival[k])
                else:
                    return self.val[k]
            k = self.link[k]

        return 0.0

    cdef safe_at(self, INT_t i, INT_t j):
        """
        Return element ``(i, j)`` but with check for out of bounds indices.

        Raises:
            IndexError: when index out of bound.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if not 0 <= i < self.nrow or not 0 <= j < self.ncol:
            raise IndexError("Index out of bounds")

        return self.at(i, j)

    def __getitem__(self, tuple key):
        """
        Return ``ll_mat[...]``.

        Args:
          key = (i,j): Must be a couple of values. Values can be:
                 * integers;
                 * slices;
                 * lists;
                 * numpy arrays

        Raises:
            IndexError: when index out of bound.

        Returns:
            If ``i`` and ``j`` are both integers, returns corresponding value ``ll_mat[i, j]``, otherwise
            returns a corresponding :class:`LLSparseMatrixView` view on the matrix.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        cdef LLSparseMatrixView view

        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            view =  MakeLLSparseMatrixView(self, <PyObject *>key[0], <PyObject *>key[1])
            return view

        cdef INT_t i = key[0]
        cdef INT_t j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    #                                            *** GET LIST ***
    cdef object _keys(self):
        """
        Return a list of tuples (i,j) of non-zero matrix entries.


        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef:
            #list list_container
            PyObject *list_p # the list that will hold the keys
            INT_t i, j, k
            INT_t pos = 0    # position in list

        if not self.is_symmetric:

            # create list
            list_p = PyList_New(self.nnz)
            if list_p == NULL:
                raise MemoryError()

            for i from 0 <= i < self.nrow:
                k = self.root[i]
                while k != -1:
                    j = self.col[k]
                    PyList_SET_ITEM(list_p, pos, Py_BuildValue("ii", i, j))
                    pos += 1
                    k = self.link[k]
        else:
            raise NotImplemented("keys() is not (yet) implemented for symmetrical LLSparseMatrix")

        return <object> list_p

    def keys(self):
        """
        Return a list of tuples ``(i,j)`` of non-zero matrix entries.

        Note:
            If ``store_zeros`` is ``True``, we zeros elements might have been stored and are returned.

        Warning:
            This method might leak memory...

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        # TODO: do we have to INCREF and DECREF???
        cdef list list_ = self._keys()
        return list_

    cdef object _values(self):
        """
        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef:
            PyObject *list_p   # the list that will hold the values
            INT_t i, k
            INT_t pos = 0        # position in list

        if not self.is_symmetric:
            list_p = PyList_New(self.nnz)
            if list_p == NULL:
                raise MemoryError()

            for i from 0<= i < self.nrow:
                k = self.root[i]
                while k != -1:
                    if self.is_complex:
                        PyList_SET_ITEM(list_p, pos, PyComplex_FromDoubles(self.val[k], self.ival[k]))
                    else:
                        PyList_SET_ITEM(list_p, pos, PyFloat_FromDouble(self.val[k]))
                    pos += 1
                    k = self.link[k]
        else:
            raise NotImplemented("values() not (yet) implemented for symmetrical LLSparseMatrix")

        return <object> list_p

    def values(self):
        """
        Return a list of the non-zero matrix entries as floats or complex if matrix is complex.

        Note:
            If ``store_zeros`` is ``True``, we zeros elements might have been stored and are returned.

        Warning:
            This method might leak memory...

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        # TODO: do we have to INCREF and DECREF???
        cdef list list_ = self._values()
        return list_

    cdef object _items(self):
        """
        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef:
            PyObject *list_p;     # the list that will hold the values
            INT_t i, j, k
            INT_t pos = 0         # position in list
            FLOAT_t val
            FLOAT_t ival

        list_p = PyList_New(self.nnz)
        if list_p == NULL:
            raise MemoryError()

        for i from 0 <= i < self.nrow:
            k = self.root[i]
            while k != -1:
                j = self.col[k]
                val = self.val[k]
                if self.is_complex:
                    ival = self.ival[k]
                    PyList_SET_ITEM(list_p, pos, Py_BuildValue("((ii)O)", i, j, PyComplex_FromDoubles(val, ival)))
                else:
                    PyList_SET_ITEM(list_p, pos, Py_BuildValue("((ii)d)", i, j, val))
                pos += 1

                k = self.link[k]

        return <object> list_p

    def items(self):
        """
        Return a list of tuples (indices, value) of the non-zero matrix entries' keys and values.

        The indices are themselves tuples (i,j) of row and column values.

        Note:
            If ``store_zeros`` is ``True``, we zeros elements might have been stored and are returned.

        Warning:
            This method might leak memory...

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        cdef list list_ = self._items()
        return list_

    ####################################################################################################################
    #                                            *** PROPERTIES ***
    property T:
        """
        {{COMPLEX: NO}}
        {{GENERIC TYPES: YES}}
        """
        def __get__(self):
            if self.is_complex:
                raise NotImplemented("This operation is not (yet) implemented for complex matrices")
            return transposed_ll_mat(self)

        def __set__(self, value):
            raise AttributeError("Transposed matrix is read-only")

        def __del__(self):
            raise AttributeError("Transposed matrix is read-only")

    ####################################################################################################################
    # Matrix conversions
    ####################################################################################################################
    def to_csr(self):
        """
        Create a corresponding CSRSparseMatrix.

        Warning:
            Memory **must** be freed by the caller!
            Column indices are **not** necessarily sorted!

        {{COMPLEX: NO}}
        {{GENERIC TYPES: YES}}
        """
        cdef INT_t * ind = <INT_t *> PyMem_Malloc((self.nrow + 1) * sizeof(INT_t))
        if not ind:
            raise MemoryError()

        cdef INT_t * col =  <INT_t*> PyMem_Malloc(self.nnz * sizeof(INT_t))
        if not col:
            raise MemoryError()

        cdef FLOAT_t * val = <FLOAT_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT_t))
        if not val:
            raise MemoryError()

        cdef FLOAT_t * ival
        if self.is_complex:
            ival = <FLOAT_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT_t))
            if not ival:
                raise MemoryError()

        cdef INT_t ind_col_index = 0  # current col index in col and val
        ind[ind_col_index] = 0

        cdef INT_t i
        cdef INT_t k

        # indices are NOT sorted for each row
        for i from 0 <= i < self.nrow:
            k = self.root[i]

            while k != -1:
                col[ind_col_index] = self.col[k]
                val[ind_col_index] = self.val[k]
                if self.is_complex:
                    ival[ind_col_index] = self.ival[k]

                ind_col_index += 1
                k = self.link[k]

            ind[i+1] = ind_col_index

        if self.is_complex:
            csr_mat = MakeCSRComplexSparseMatrix(nrow=self.nrow, ncol=self.ncol, nnz=self.nnz, ind=ind, col=col, val=val, ival=ival)
        else:
            csr_mat = MakeCSRSparseMatrix(nrow=self.nrow, ncol=self.ncol, nnz=self.nnz, ind=ind, col=col, val=val)

        return csr_mat

    def to_csc(self):
        """
        Create a corresponding CSCSparseMatrix.

        Warning:
            Memory **must** be freed by the caller!
            Column indices are **not** necessarily sorted!

        {{COMPLEX: NO}}
        {{GENERIC TYPES: YES}}
        """
        if self.is_complex:
            raise NotImplemented("This operation is not (yet) implemented for complex matrices")

        cdef INT_t * ind = <INT_t *> PyMem_Malloc((self.ncol + 1) * sizeof(INT_t))
        if not ind:
            raise MemoryError()

        cdef INT_t * row = <INT_t *> PyMem_Malloc(self.nnz * sizeof(INT_t))
        if not row:
            raise MemoryError()

        cdef FLOAT_t * val = <FLOAT_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT_t))
        if not val:
            raise MemoryError()


        cdef:
            INT_t i, k


        # start by collecting the number of rows for each column
        # this is to create the ind vector but not only...
        cdef INT_t * col_indexes = <INT_t *> calloc(self.ncol + 1, sizeof(INT_t))
        if not ind:
            raise MemoryError()

        col_indexes[0] = 0

        for i from 0 <= i < self.nrow:
            k = self.root[i]
            while k != -1:
                col_indexes[self.col[k] + 1] += 1
                k = self.link[k]

        # ind
        for i from 1 <= i <= self.ncol:
            col_indexes[i] = col_indexes[i - 1] + col_indexes[i]

        memcpy(ind, col_indexes, (self.ncol + 1) * sizeof(INT_t) )
        assert ind[self.ncol] == self.nnz

        # row and val
        # we have ind: we know exactly where to put the row indices for each column
        # we use col_indexes to get the next index in row and val
        for i from 0 <= i < self.nrow:
            k = self.root[i]
            while k != -1:
                col_index = col_indexes[self.col[k]]
                row[col_index] = i
                val[col_index] = self.val[k]
                col_indexes[self.col[k]] += 1 # update index in row and val

                k = self.link[k]


        free(col_indexes)

        csc_mat = MakeCSCSparseMatrix(nrow=self.nrow, ncol=self.ncol, nnz=self.nnz, ind=ind, row=row, val=val)

        return csc_mat

    def to_csb(self):
        """
        Create a corresponding CSBSparseMatrix.

        Warning:
            Memory **must** be freed by the caller!
            Column indices are **not** necessarily sorted!

        {{COMPLEX: NO}}
        {{GENERIC TYPES: NO}}
        """
        raise NotImplemented("This operation is not (yet) implemented")

    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def __mul__(self, B):
        """

        {{COMPLEX: NO}}
        {{GENERIC TYPES: YES}}
        """
        if self.is_complex:
            raise NotImplemented("This operation is not (yet) implemented for complex matrices")

        # CASES
        if isinstance(B, LLSparseMatrix):
            return multiply_two_ll_mat(self, B)
        elif isinstance(B, np.ndarray):
            # test type
            assert B.dtype == np.float64, "Multiplication only allowed with an array of C-doubles (numpy float64)!"

            if B.ndim == 2:
                return multiply_ll_mat_with_numpy_ndarray(self, B)
            elif B.ndim == 1:
                return multiply_ll_mat_with_numpy_vector2(self, B)
            else:
                raise IndexError("Matrix dimensions must agree")
        else:
            raise NotImplemented("Multiplication with this kind of object not implemented yet...")

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def print_to(self, OUT):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        # TODO: don't create temporary matrix
        cdef INT_t i, k, first = 1

        print(self._matrix_description_before_printing(), file=OUT)

        cdef FLOAT_t *mat
        cdef INT_t j
        cdef FLOAT_t val, ival

        if not self.nnz:
            return

        if self.is_complex:
            if self.nrow <= LL_MAT_COMPLEX_PPRINT_COL_THRESH and self.ncol <= LL_MAT_COMPLEX_PPRINT_ROW_THRESH:
                # create linear vector presentation
                # TODO: put in a method of its own
                mat = <FLOAT_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(FLOAT_t) * 2)

                if not mat:
                    raise MemoryError()

                for i from 0 <= i < self.nrow:
                    for j from 0 <= j < self.ncol:
                        mat[(i* 2 * self.ncol) + (2*j)] = 0.0
                        mat[(i* 2 *self.ncol) + (2*j) + 1] = 0.0
                    k = self.root[i]
                    while k != -1:
                        mat[i* 2 * self.ncol + self.col[k] * 2] = self.val[k]
                        mat[i* 2 * self.ncol + self.col[k] * 2 + 1] = self.ival[k]
                        k = self.link[k]

                for i from 0 <= i < self.nrow:
                    for j from 0 <= j < self.ncol:
                        val = mat[i* 2 * self.ncol + j*2]
                        ival = mat[i* 2 * self.ncol + j*2 + 1]
                        #print('%9.*f ' % (6, val), file=OUT, end='')
                        print('({0:9.6f} {1:9.6f}) '.format(val, ival), end='', file=OUT)
                    print(file=OUT)

                PyMem_Free(mat)
            else:
                print('Matrix too big to print out', file=OUT)
        else:
            if self.nrow <= LL_MAT_PPRINT_COL_THRESH and self.ncol <= LL_MAT_PPRINT_ROW_THRESH:
                # create linear vector presentation


                # TODO: put in a method of its own
                mat = <FLOAT_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(FLOAT_t))

                if not mat:
                    raise MemoryError()

                #for i in xrange(self.nrow):
                for i from 0 <= i < self.nrow:
                    #for j in xrange(self.ncol):
                    for j from 0 <= j < self.ncol:
                        mat[i* self.ncol + j] = 0.0
                    k = self.root[i]
                    while k != -1:
                        mat[(i*self.ncol)+self.col[k]] = self.val[k]
                        k = self.link[k]

                #for i in xrange(self.nrow):
                for i from 0 <= i < self.nrow:
                    #for j in xrange(self.ncol):
                    for j from 0 <= j < self.ncol:
                        val = mat[(i*self.ncol)+j]
                        #print('%9.*f ' % (6, val), file=OUT, end='')
                        print('{0:9.6f} '.format(val), end='', file=OUT)
                    print(file=OUT)

                PyMem_Free(mat)

            else:
                print('Matrix too big to print out', file=OUT)

    def save_to(self, filename, format):
        if format == "MM":
            MakeMMFileFromSparseMatrix(mm_filename=filename, A=self)



include "ll_mat_details/ll_mat_multiplication.pxi"

########################################################################################################################
# Common matrix operations
########################################################################################################################
cdef bint PyLLSparseMatrix_Check(object obj):
    """
    {{COMPLEX: YES}}
    {{GENERIC TYPES: YES}}
    """
    return isinstance(obj, LLSparseMatrix)

include "ll_mat_details/ll_mat_transpose.pxi"

########################################################################################################################
# Assignments
########################################################################################################################
include "ll_mat_details/ll_mat_assignment.pxi"



include "ll_mat_details/ll_mat_real_assignment_kernels.pxi"

include "ll_mat_details/ll_mat_real_multiplication_kernels.pxi"

########################################################################################################################
# Factory methods
########################################################################################################################
def MakeLLSparseMatrix(**kwargs):
    """
    TEMPORARY function...

    Args:

    {{COMPLEX: NO}}
    {{GENERIC TYPES: NO}}
    """
    # TODO: rewrite function!!!
    # TODO: add symmetrical case
    cdef INT_t nrow = kwargs.get('nrow', -1)
    cdef INT_t ncol = kwargs.get('ncol', -1)
    cdef INT_t size = kwargs.get('size', -1)
    cdef INT_t size_hint = kwargs.get('size_hint', LL_MAT_DEFAULT_SIZE_HINT)
    cdef bint store_zeros = kwargs.get('store_zeros', False)
    cdef bint is_symmetric = kwargs.get('is_symmetric', False)
    cdef bint is_complex = kwargs.get('is_complex', False)
    cdef bint test_bounds = kwargs.get('test_bounds', True)

    matrix = kwargs.get('matrix', None)

    mm_filename = kwargs.get('mm_filename', None)

    cdef INT_t real_nrow
    cdef INT_t real_ncol

    # CASE 1
    if matrix is None and mm_filename is None:
        if nrow != -1 and ncol != -1:
            if size != -1:
                assert nrow == ncol == size, "Mismatch between nrow, ncol and size"
            real_nrow = nrow
            real_ncol = ncol
        elif nrow != -1 and ncol == -1:
            if size != -1:
                assert size == nrow, "Mismatch between nrow and size"
            real_nrow = nrow
            real_ncol = nrow
        elif nrow == -1 and ncol != -1:
            if size != -1:
                assert ncol == size, "Mismatch between ncol and size"
            real_nrow = ncol
            real_ncol = ncol
        else:
            assert size != -1, "No size given"
            real_nrow = size
            real_ncol = size

        return LLSparseMatrix(nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)

    # CASE 2
    cdef FLOAT_t[:, :] matrix_view
    cdef INT_t i, j
    cdef FLOAT_t value

    if matrix is not None and mm_filename is None:
        # TODO: direct access into the numpy array
        # TODO: skip views... ?
        # TODO: take into account the complex case!
        if len(matrix.shape) != 2:
            raise IndexError('Matrix must be of dimension 2 (not %d)' % len(matrix.shape))

        matrix_view = matrix

        if nrow != -1:
            if nrow != matrix.shape[0]:
                raise IndexError('nrow (%d) doesn\'t match matrix row count' % nrow)

        if ncol != -1:
            if ncol != matrix.shape[1]:
                raise IndexError('ncol (%d) doesn\'t match matrix col count' % ncol)

        nrow = matrix.shape[0]
        ncol = matrix.shape[1]


        ll_mat = LLSparseMatrix(nrow=nrow, ncol=ncol, size_hint=size_hint)

        #for i in xrange(nrow):
        for i from 0 <= i < nrow:
            #for j in xrange(ncol):
            for j from 0 <= j < ncol:
                value = matrix_view[i, j]
                if value != 0.0:
                    ll_mat[i, j] = value

        return ll_mat

    if mm_filename is not None:
        return MakeLLSparseMatrixFromMMFile2(mm_filename=mm_filename, store_zeros=store_zeros, test_bounds=test_bounds)



