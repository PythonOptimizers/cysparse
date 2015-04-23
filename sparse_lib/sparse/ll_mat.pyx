"""
ll_mat extension.


"""
from __future__ import print_function

from sparse_lib.sparse.sparse_mat cimport MutableSparseMatrix
from sparse_lib.sparse.csr_mat cimport MakeCSRSparseMatrix
from sparse_lib.sparse.csc_mat cimport MakeCSCSparseMatrix
#from sparse_lib.utils.equality cimport values_are_equal
from sparse_lib.sparse.IO.mm cimport MakeLLSparseMatrixFromMMFile


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


cdef int LL_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default
cdef double LL_MAT_INCREASE_FACTOR = 1.5      # reallocating factor if size is not enough, must be > 1
cdef int LL_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef int LL_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format

#include 'll_mat_slices.pxi'

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
    #cdef:
    #    int     free      # index to first element in free chain
    #    double *val       # pointer to array of values
    #    int    *col       # pointer to array of indices, see doc
    #    int    *link      # pointer to array of indices, see doc
    #    int    *root      # pointer to array of indices, see doc

    def __cinit__(self, int nrow, int ncol, int size_hint=LL_MAT_DEFAULT_SIZE_HINT, bint is_symmetric=False, bint store_zeros=False):

        if size_hint < 1:
            raise ValueError('size_hint (%d) must be >= 1' % size_hint)

        val = <double *> PyMem_Malloc(self.size_hint * sizeof(double))
        if not val:
            raise MemoryError()
        self.val = val

        col = <int *> PyMem_Malloc(self.size_hint * sizeof(int))
        if not col:
            raise MemoryError()
        self.col = col

        link = <int *> PyMem_Malloc(self.size_hint * sizeof(int))
        if not link:
            raise MemoryError()
        self.link = link

        root = <int *> PyMem_Malloc(self.nrow * sizeof(int))
        if not root:
            raise MemoryError()
        self.root = root

        self.nalloc = self.size_hint
        self.free = -1

        cdef int i
        for i from 0 <= i < nrow:
            root[i] = -1

    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.col)
        PyMem_Free(self.link)
        PyMem_Free(self.root)

    cdef _realloc(self, int nalloc_new):
        """
        Realloc space for the internal arrays.

        Note:
            Internal arrays can be expanded or shrunk.

        """
        cdef:
            void *temp
            #int nalloc_new

        temp = <int *> PyMem_Realloc(self.col, nalloc_new * sizeof(int))
        if not temp:
            raise MemoryError()
        self.col = <int*>temp

        temp = <int *> PyMem_Realloc(self.link, nalloc_new * sizeof(int))
        if not temp:
            raise MemoryError()
        self.link = <int *>temp

        temp = <double *> PyMem_Realloc(self.val, nalloc_new * sizeof(double))
        if not temp:
            raise MemoryError()
        self.val = <double *>temp

        self.nalloc = nalloc_new

    cdef _realloc_expand(self):
        """
        Realloc space for internal arrays.

        Note:
            We use ``LL_MAT_INCREASE_FACTOR`` as expanding factor.
        """
        assert LL_MAT_INCREASE_FACTOR > 1.0
        cdef int real_new_alloc = <int>(<double>LL_MAT_INCREASE_FACTOR * self.nalloc) + 1

        return self._realloc(real_new_alloc)

    def compress(self):
        """
        Shrink matrix to its minimal size.
        """
        cdef:
            #double *val;
            #int *col, *link;
            int i, k, k_next, k_last, k_new, nalloc_new;

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
            The exact number of bits used to store the matrix.
        """
        cdef int total_memory = 0

        # root
        total_memory += self.nrow * sizeof(int)
        # col
        total_memory += self.nalloc * sizeof(int)
        # link
        total_memory += self.nalloc * sizeof(int)
        # val
        total_memory += self.nalloc * sizeof(double)

        return total_memory

    ####################################################################################################################
    # Set/Get items
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    cdef put(self, int i, int j, double value):
        """
        Set ``A[i, j] = value`` directly.

        Note:
            Store zero elements **only** if ``store_zeros`` is ``True``.

        Warning:
            No out of bound check.

        See:
            :meth:`safe_put`.

        """
        if self.is_symmetric and i < j:
            raise IndexError('Write operation to upper triangle of symmetric matrix not allowed')

        cdef int k, new_elem, last, col

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
        #if value != 0.0 or self.store_zeros
        if self.store_zeros or value != 0.0: #not values_are_equal(value, 0.0):
            if col == j:
                # element already exist
                self.val[k] = value
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
                self.col[new_elem] = j
                self.link[new_elem] = k

                if last == -1:
                    self.root[i] = new_elem
                else:
                    self.link[last] = new_elem

                self.nnz += 1

        else:
            # value == 0.0: if element exists but we don't store zero elements
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

    cdef safe_put(self, int i, int j, double value):
        """
        Set ``A[i, j] = value`` directly.

        Raises:
            IndexError: when index out of bound.
        """
        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            raise IndexError('Indices out of range')

        self.put(i, j, value)

    cdef assign(self, LLSparseMatrixView view, object obj):
        # test if view correspond...
        assert self == view.A

        update_ll_mat_matrix_from_c_arrays_indices_assign(self, view.row_indices, view.nrow,
                                                       view.col_indices, view.ncol, obj)

    def put_triplet(self,  list val, list index_i, list index_j):
        cdef index_i_length = len(index_i)
        cdef index_j_length = len(index_j)
        cdef val_length = len(val)

        assert index_j_length == index_j_length == val_length, "All lists must be of equal length"

        cdef Py_ssize_t i

        for i from 0 <= i < index_i_length:
            self.safe_put(index_i[i], index_j[i], val[i])

    def __setitem__(self, tuple key, value):
        """
        A[i, j] = value

        Raises:
            IndexError: when index out of bound.

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

        cdef int i = key[0]
        cdef int j = key[1]

        self.safe_put(i, j, <double> value)

    ####################################################################################################################
    #                                            *** GET ***
    cdef at(self, int i, int j):
        """
        Return element ``(i, j)``.

        Warning:
            There is not out of bounds test.

        See:
            :meth:`safe_at`.

        """
        cdef int k, t

        if self.is_symmetric and i < j:
            t = i; i = j; j = t

        k = self.root[i]

        while k != -1:
            if self.col[k] == j:
                return self.val[k]
            k = self.link[k]

        return 0.0

    cdef safe_at(self, int i, int j):
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
        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        cdef LLSparseMatrixView view

        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            view =  MakeLLSparseMatrixView(self, <PyObject *>key[0], <PyObject *>key[1])
            return view

        cdef int i = key[0]
        cdef int j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    #                                            *** GET LIST ***
    cdef object _keys(self):
        """
        Return a list of tuples (i,j) of non-zero matrix entries.


        """
        cdef:
            #list list_container
            PyObject *list_p # the list that will hold the keys
            int i, j, k
            int pos = 0    # position in list

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

        Warning:
            This method might leak memory...
        """
        # TODO: do we have to INCREF and DECREF???
        cdef list list_ = self._keys()
        return list_

    cdef object _values(self):
        cdef:
            PyObject *list_p   # the list that will hold the values
            int i, k
            int pos = 0        # position in list

        if not self.is_symmetric:
            list_p = PyList_New(self.nnz)
            if list_p == NULL:
                raise MemoryError()

            for i from 0<= i < self.nrow:
                k = self.root[i]
                while k != -1:
                    PyList_SET_ITEM(list_p, pos, PyFloat_FromDouble(self.val[k]))
                    pos += 1
                    k = self.link[k]
        else:
            raise NotImplemented("values() not (yet) implemented for symmetrical LLSparseMatrix")

        return <object> list_p

    def values(self):
        """
        Return a list of the non-zero matrix entries as floats.

        Warning:
            This method might leak memory...
        """
        # TODO: do we have to INCREF and DECREF???
        cdef list list_ = self._values()
        return list_

    cdef object _items(self):
        cdef:
            PyObject *list_p;     # the list that will hold the values
            int i, j, k
            int pos = 0         # position in list
            double val

        list_p = PyList_New(self.nnz)
        if list_p == NULL:
            raise MemoryError()

        for i from 0 <= i < self.nrow:
            k = self.root[i]
            while k != -1:
                j = self.col[k]
                val = self.val[k]
                PyList_SET_ITEM(list_p, pos, Py_BuildValue("((ii)d)", i, j, val))
                pos += 1

                k = self.link[k]

        return <object> list_p

    def items(self):
        """
        Return a list of tuples (indices, value) of the non-zero matrix entries' keys and values.

        The indices are themselves tuples (i,j) of row\n\
        and column values.

        Warning:
            This method might leak memory...
        """
        cdef list list_ = self._items()
        return list_

    ####################################################################################################################
    #                                            *** PROPERTIES ***
    property T:
        def __get__(self):
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
        """

        cdef int * ind = <int *> PyMem_Malloc((self.nrow + 1) * sizeof(int))
        if not ind:
            raise MemoryError()

        cdef int * col =  <int*> PyMem_Malloc(self.nnz * sizeof(int))
        if not col:
            raise MemoryError()

        cdef double * val = <double *> PyMem_Malloc(self.nnz * sizeof(double))
        if not val:
            raise MemoryError()

        cdef int ind_col_index = 0  # current col index in col and val
        ind[ind_col_index] = 0

        cdef int i
        cdef int k

        # indices are NOT sorted for each row
        for i from 0 <= i < self.nrow:
            k = self.root[i]

            while k != -1:
                col[ind_col_index] = self.col[k]
                val[ind_col_index] = self.val[k]

                ind_col_index += 1
                k = self.link[k]

            ind[i+1] = ind_col_index

        csr_mat = MakeCSRSparseMatrix(nrow=self.nrow, ncol=self.ncol, nnz=self.nnz, ind=ind, col=col, val=val)

        return csr_mat

    def to_csc(self):
        """
        Create a corresponding CSCSparseMatrix.

        Warning:
            Memory **must** be freed by the caller!
            Column indices are **not** necessarily sorted!
        """
        cdef int * ind = <int *> PyMem_Malloc((self.ncol + 1) * sizeof(int))
        if not ind:
            raise MemoryError()

        cdef int * row = <int *> PyMem_Malloc(self.nnz * sizeof(int))
        if not row:
            raise MemoryError()

        cdef double * val = <double *> PyMem_Malloc(self.nnz * sizeof(double))
        if not val:
            raise MemoryError()


        cdef:
            int i, k


        # start by collecting the number of rows for each column
        # this is to create the ind vector but not only...
        cdef int * col_indexes = <int *> calloc(self.ncol + 1, sizeof(int))
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

        memcpy(ind, col_indexes, (self.ncol + 1) * sizeof(int) )
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
        """
        pass

    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def __mul__(self, B):
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
    def __repr__(self):
        s = "LLSparseMatrix of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
        return s

    def print_to(self, OUT):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.
        """
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        cdef int i, k, first = 1
        symmetric_str = None

        if self.is_symmetric:
            symmetric_str = 'symmetric'
        else:
            symmetric_str = 'general'

        print('LLSparseMatrix (%s, [%d,%d]):' % (symmetric_str, self.nrow, self.ncol), file=OUT)

        cdef double *mat
        cdef int j
        cdef double val

        if not self.nnz:
            return

        if self.nrow <= LL_MAT_PPRINT_COL_THRESH and self.ncol <= LL_MAT_PPRINT_ROW_THRESH:
            # create linear vector presentation
            # TODO: put in a method of its own
            mat = <double *> PyMem_Malloc(self.nrow * self.ncol * sizeof(double))

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
                    print('{0:9.6f} '.format(val), end='')
                print()

            PyMem_Free(mat)


########################################################################################################################
# Factory methods
########################################################################################################################
def MakeLLSparseMatrix(**kwargs):
    """
    TEMPORARY function...

    Args:


    """
    # TODO: rewrite function!!!
    # TODO: add symmetrical case
    cdef int nrow = kwargs.get('nrow', -1)
    cdef int ncol = kwargs.get('ncol', -1)
    cdef int size = kwargs.get('size', -1)
    cdef int size_hint = kwargs.get('size_hint', LL_MAT_DEFAULT_SIZE_HINT)
    cdef bint store_zeros = kwargs.get('store_zeros', False)
    cdef bint is_symmetric = kwargs.get('is_symmetrix', False)

    matrix = kwargs.get('matrix', None)

    mm_filename = kwargs.get('mm_filename', None)

    cdef int real_nrow
    cdef int real_ncol

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

        return LLSparseMatrix(nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)

    # CASE 2
    cdef double[:, :] matrix_view
    cdef int i, j
    cdef double value

    if matrix is not None and mm_filename is None:
        # TODO: direct access into the numpy array
        # TODO: skip views... ?
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
        return MakeLLSparseMatrixFromMMFile(mm_filename)

########################################################################################################################
# Multiplication functions
########################################################################################################################
cdef LLSparseMatrix multiply_two_ll_mat(LLSparseMatrix A, LLSparseMatrix B):
    """
    Multiply two :class:`LLSparseMatrix` ``A`` and ``B``.

    Args:
        A: An :class:``LLSparseMatrix`` ``A``.
        B: An :class:``LLSparseMatrix`` ``B``.

    Returns:
        A **new** :class:``LLSparseMatrix`` ``C = A * B``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplemented``: When matrix ``A`` or ``B`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    # TODO: LLSparseMatrix * A, LLSparseMatrix * B ...
    # test dimensions
    cdef int A_nrow = A.nrow
    cdef int A_ncol = A.ncol

    cdef int B_nrow = B.nrow
    cdef int B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef int C_nrow = A_nrow
    cdef int C_ncol = B_ncol

    cdef bint store_zeros = A.store_zeros and B.store_zeros
    cdef int size_hint = A.size_hint

    C = LLSparseMatrix(nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zeros=store_zeros)


    # CASES
    if not A.is_symmetric and not B.is_symmetric:
        pass
    else:
        raise NotImplemented("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    cdef:
        double valA
        int iA, jA, kA, kB

    for iA from 0 <= iA < A_nrow:
        kA = A.root[iA]

        while kA != -1:
            valA = A.val[kA]
            jA = A.col[kA]
            kA = A.link[kA]

            # add jA-th row of B to iA-th row of C
            kB = B.root[jA]
            while kB != -1:
                update_ll_mat_item_add(C, iA, B.col[kB], valA*B.val[kB])
                kB = B.link[kB]
    return C


cdef multiply_ll_mat_with_numpy_ndarray(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=2] B):
    raise NotImplemented("Multiplication with numpy ndarray of dim 2 not implemented yet")

cdef cnp.ndarray[cnp.double_t, ndim=1] multiply_ll_mat_with_numpy_vector(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=1, mode="c"] b):
    """
    Multiply a :class:`LLSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`LLSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        ``c = A * b``: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    """
    # TODO: take strides into account!
    # test if numpy array is c-contiguous

    cdef int A_nrow = A.nrow
    cdef int A_ncol = A.ncol

    #temp = cnp.NPY_DOUBLE

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef double * b_data = <double *> b.data

    # array c = A * b
    cdef cnp.ndarray[cnp.double_t, ndim=1] c = np.empty(A_nrow, dtype=np.float64)
    cdef double * c_data = <double *> c.data

    cdef:
        int i, j
        int k

        double val
        double val_c

    for i from 0 <= i < A_nrow:
        k = A.root[i]

        val_c = 0.0

        while k != -1:
            val = A.val[k]
            j = A.col[k]
            k = A.link[k]

            val_c += val * b_data[j]

        c_data[i] = val_c


    return c


cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] multiply_ll_mat_with_numpy_vector2(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=1] b):
    """
    Multiply a :class:`LLSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`LLSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        ``c = A * b``: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is more general as it takes into account strides in the numpy arrays and if the :class:`LLSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef int A_nrow = A.nrow
    cdef int A_ncol = A.ncol

    cdef size_t sd = sizeof(double)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef double * b_data = <double *> b.data

    # array c = A * b
    cdef cnp.ndarray[cnp.double_t, ndim=1] c = np.empty(A_nrow, dtype=np.float64)
    cdef double * c_data = <double *> c.data

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.is_symmetric:
            multiply_sym_ll_mat_with_numpy_vector_kernel(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_numpy_vector_kernel(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
    else:
        if A.is_symmetric:
            multiply_sym_ll_mat_with_strided_numpy_vector_kernel(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_strided_numpy_vector_kernel(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.link, A.root)

    return c


########################################################################################################################
# Common matrix operations
########################################################################################################################
cdef LLSparseMatrix transposed_ll_mat(LLSparseMatrix A):
    """
    Compute transposed matrix.

    Args:
        A: A :class:`LLSparseMatrix` :math:`A`.

    Note:
        The transposed matrix uses the same amount of internal memory as the

    Returns:
        The corresponding transposed :math:`A^t` :class:`LLSparseMatrix`.
    """
    # TODO: optimize to pure Cython code
    if A.is_symmetric:
        raise NotImplemented("Transposed is not implemented yet for symmetric matrices")

    cdef:
        int A_nrow = A.nrow
        int A_ncol = A.ncol

        int At_nrow = A.ncol
        int At_ncol = A.nrow

        int At_nalloc = A.nalloc

        int i, k
        double val

    cdef LLSparseMatrix transposed_A = LLSparseMatrix(nrow =At_nrow, ncol=At_ncol, size_hint=At_nalloc)

    for i from 0 <= i < A_nrow:
        k = A.root[i]

        while k != -1:
            val = A.val[k]
            j = A.col[k]
            k = A.link[k]

            transposed_A[j, i] = val


    return transposed_A


########################################################################################################################
# Assignments
########################################################################################################################
cdef int PyLLSparseMatrix_Check(object obj):
    return isinstance(obj, LLSparseMatrix)

cdef update_ll_mat_matrix_from_c_arrays_indices_assign(LLSparseMatrix A, int * index_i, Py_ssize_t index_i_length,
                                                       int * index_j, Py_ssize_t index_j_length, object obj):
    """
    Update-assign (sub-)matrix: A[..., ...] = obj.

    Args:
        A: An :class:`LLSparseMatrix` object.
        index_i: C-arrays with ``int`` indices.
        index_i_length: Length of ``index_i``.
        index_j: C-arrays with ``int`` indices.
        index_j_length: Length of ``index_j``.
        obj: Any Python object that implements ``__getitem__()`` and accepts a ``tuple`` ``(i, j)``.

    Warning:
        There are not test whatsoever.
    """
    cdef:
        Py_ssize_t i
        Py_ssize_t j

    # TODO: use internal arrays like triplet (i, j, val)?
    # but indices can be anything...
    if PyLLSparseMatrix_Check(obj):
        #ll_mat =  obj
        for i from 0 <= i < index_i_length:
            for j from 0 <= j < index_j_length:
                A.put(index_i[i], index_j[j], obj[i, j])

    else:
        for i from 0 <= i < index_i_length:
            for j from 0 <= j < index_j_length:
                A.put(index_i[i], index_j[j], <double> obj[tuple(i, j)]) # not really optimized...

cdef bint update_ll_mat_item_add(LLSparseMatrix A, int i, int j, double x):
    """
    Update-add matrix entry: ``A[i,j] += x``

    Args:
        A: Matrix to update.
        i, j: Coordinates of item to update.
        x (double): Value to add to item to update ``A[i, j]``.

    Returns:
        True.

    Raises:
        ``IndexError`` when non writing to lower triangle of a symmetric matrix.
    """
    cdef:
        int k, new_elem, col, last

    if A.is_symmetric and i < j:
        raise IndexError("Write operation to upper triangle of symmetric matrix not allowed")

    if not A.store_zeros and x == 0.0:
        return True

    # Find element to be updated
    col = last = -1
    k = A.root[i]
    while k != -1:
        col = A.col[k]
        if col >= j:
            break
        last = k
        k = A.link[k]

    if col == j:
        # element already exists: compute updated value
        x += A.val[k]

        if A.store_zeros and x == 0.0:
            #  the updated element is zero and must be removed

            # relink row i
            if last == -1:
                A.root[i] = A.link[k]
            else:
                A.link[last] = A.link[k]

            # add element to free list
            A.link[k] = A.free
            A.free = k

            A.nnz -= 1
        else:
            A.val[k] = x
    else:
        # new item
        if A.free != -1:
            # use element from the free chain
            new_elem = A.free
            A.free = A.link[new_elem]
        else:
            # append new element to the end
            new_elem = A.nnz

            # test if there is space for a new element
            if A.nnz == A.nalloc:
                A._realloc_expand()

        A.val[new_elem] = x
        A.col[new_elem] = j
        A.link[new_elem] = k
        if last == -1:
            A.root[i] = new_elem
        else:
            A.link[last] = new_elem
        A.nnz += 1

    return True


########################################################################################################################
# Matrix - vector multiplication kernels
########################################################################################################################
# C-contiguous, no symmetric
cdef void multiply_ll_mat_with_numpy_vector_kernel(int m, double *x, double *y,
         double *val, int *col, int *link, int *root):
    """
    Compute ``y = A * x``.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        double s
        int i, k

    for i from 0 <= i < m:
        s = 0.0
        k = root[i]

        while k != -1:
          s += val[k] * x[col[k]]
          k = link[k]

        y[i] = s

# C-contiguous, symmetric
cdef void multiply_sym_ll_mat_with_numpy_vector_kernel(int m, double *x, double *y,
             double *val, int *col, int *link, int *root):
    """
    Compute ``y = A * x``.

    ``A`` is a **symmetric** :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        double s, v, xi
        int i, j, k

    for i from 0 <= i < m:
        xi = x[i]
        s = 0.0
        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j]
            if i != j:
                y[j] += v * xi
            k = link[k]

        y[i] = s

# Non C-contiguous, non symmetric
cdef void multiply_ll_mat_with_strided_numpy_vector_kernel(int m,
            double *x, int incx,
            double *y, int incy,
            double *val, int *col, int *link, int *root):
    """
    Compute ``y = A * x``.

    ``A`` is :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional **non** C-contiguous numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider *both* numpy arrays as **non** C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        double s
        int i, k

    for i from 0 <= i < m:
        s = 0.0
        k = root[i]

        while k != -1:
            s += val[k] * x[col[k]*incx]
            k = link[k]

        y[i*incy] = s

# Non C-contiguous, non symmetric
cdef void multiply_sym_ll_mat_with_strided_numpy_vector_kernel(int m,
                double *x, int incx,
                double *y, int incy,
                double *val, int *col, int *link, int *root):
    """
    Compute ``y = A * x``.

    ``A`` is a **symmetric** :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional **non** C-contiguous numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider *both* numpy arrays as **non** C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        double s, v, xi
        int i, j, k

    for i from 0 <= i < m:
        xi = x[i*incx]
        s = 0.0
        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j*incx]
            if i != j:
                y[j*incy] += v * xi
            k = link[k]

        y[i*incy] = s
