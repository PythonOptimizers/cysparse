"""
ll_mat extension.


"""
from __future__ import print_function

from sparse_lib.sparse.sparse_mat cimport MutableSparseMatrix
from sparse_lib.sparse.csr_mat cimport CSRSparseMatrix, MakeCSRSparseMatrix



# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

cnp.import_array()

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc,free
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

    def __cinit__(self, int nrow, int ncol, int size_hint=LL_MAT_DEFAULT_SIZE_HINT, bint store_zeros=False):

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

    cdef _realloc(self):
        """
        Realloc space for the 1D arrays.

        Warning:
            1D arrays can only be expanded with this method.

        """
        cdef:
            void *temp
            int nalloc_new

        # we have to reallocate some space
        # increase size of col, val and link arrays
        assert LL_MAT_INCREASE_FACTOR > 1.0
        nalloc_new = <int>(<double>LL_MAT_INCREASE_FACTOR * self.nalloc) + 1

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

    ####################################################################################################################
    # Get/Set items
    ####################################################################################################################
    def _assert_length_tuple_is_2(self, tuple key):
        """
        Assert that length of tuple is 2.

        Raises:
            ``IndexError`` if the length of tuple is **not** 2.
        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

    def _assert_indexed_tuple_is_within_bounds(self, tuple key):
        """
        Assert that both integers in the tuple ``key = (i, j)`` are within bounds.

        Raises:
            ``IndexError`` if one of the index is out of bound.
        """
        cdef int i = key[0]
        cdef int j = key[1]

        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            raise IndexError('Indices out of range')

    def __setitem__(self, tuple key, double value):
        self._assert_length_tuple_is_2(key)

        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[0]):
            raise NotImplemented("Assignment with non integer indices is not implemented yet")

        # both element of the tuple **are** integers
        self._assert_indexed_tuple_is_within_bounds(key)

        cdef int i = key[0]
        cdef int j = key[1]

        if self.is_symmetric and i < j:
            raise IndexError('Write operation to upper triangle of symmetric matrix')

        cdef void *temp
        cdef int k, new_elem, last, col

        cdef int nalloc_new

        # Find element to be set (or removed)
        col = last = -1
        k = self.root[i]
        while k != -1:
            col = self.col[k]
            if col >= j:
                break
            last = k
            k = self.link[k]

        if value != 0.0 or self.store_zeros:
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
                    self._realloc()

                self.val[new_elem] = value
                self.col[new_elem] = j
                self.link[new_elem] = k

                if last == -1:
                    self.root[i] = new_elem
                else:
                    self.link[last] = new_elem

                self.nnz += 1
        else:
            # value == 0.0
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

    def __getitem__(self, tuple key):
        """
        Return ``ll_mat[...]``.

        Args:
          key = (i,j): Must be a couple of values. Values can be:
                 * integers;
                 * lists;
                 * numpy arrays

        Returns:
            If ``i`` and ``j`` are both integers, return corresponding value ``ll_mat[i, j]``, otherwise
            return the corresponding :class:`LLSparseMatrixView`.
        """
        self._assert_length_tuple_is_2(key)

        cdef LLSparseMatrixView view

        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[0]):
            view =  MakeLLSparseMatrixView(self, <PyObject *>key[0], <PyObject *>key[1])
            return view

        # both element of the tuple **are** integers
        self._assert_indexed_tuple_is_within_bounds(key)

        cdef int i = key[0]
        cdef int j = key[1]

        cdef int k, t

        if self.is_symmetric and i < j:
            t = i; i = j; j = t

        k = self.root[i]

        while k != -1:
            if self.col[k] == j:
                return self.val[k]
            k = self.link[k]

        return 0.0

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
        #for i in xrange(self.nrow):
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
                return multiply_ll_mat_with_numpy_vector(self, B)
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
    cdef int nrow = kwargs.get('nrow', -1)
    cdef int ncol = kwargs.get('ncol', -1)
    cdef int size_hint = kwargs.get('size_hint', LL_MAT_DEFAULT_SIZE_HINT)
    matrix = kwargs.get('matrix', None)

    # CASE 1
    if matrix is None and nrow != -1 and ncol != -1:
        return LLSparseMatrix(nrow=nrow, ncol=ncol, size_hint=size_hint)

    # CASE 2
    cdef double[:, :] matrix_view
    cdef int i, j
    cdef double value

    if matrix is not None:
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

########################################################################################################################
# Multiplication functions
########################################################################################################################
cdef LLSparseMatrix multiply_two_ll_mat(LLSparseMatrix A, LLSparseMatrix B):
    """
    Multiply two :class:`LLSparseMatrix` ``A`` and ``B``.

    Args:
        A: A :class:``LLSparseMatrix`` ``A``.
        B: A :class:``LLSparseMatrix`` ``B``.

    Returns:
        A **new** :class:``LLSparseMatrix`` ``C = A * B``.

    Raises:
        ``NotImplemented``: When matrix ``A`` or ``B`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
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

cdef cnp.ndarray[cnp.double_t, ndim=1] multiply_ll_mat_with_numpy_vector(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=1] b):
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
    cdef int A_nrow = A.nrow
    cdef int A_ncol = A.ncol

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
    # TODO: optimized to pure Cython code
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
        ``IndexError`` when writing to lower triangle of a symmetric matrix.
    """
    cdef:
        int k, new_elem, col, last

    if A.is_symmetric and i < j:
        raise IndexError("Write operation to lower triangle of symmetric matrix (only fill in upper triangle for symmetric matrices)")

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
                A._realloc()

        A.val[new_elem] = x
        A.col[new_elem] = j
        A.link[new_elem] = k
        if last == -1:
            A.root[i] = new_elem
        else:
            A.link[last] = new_elem
        A.nnz += 1

    return True


