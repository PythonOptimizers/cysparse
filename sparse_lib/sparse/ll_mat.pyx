"""
ll_mat extension.


"""
from __future__ import print_function

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


cdef int LL_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default
cdef double LL_MAT_INCREASE_FACTOR = 1.5      # reallocating factor if size is not enough, must be > 1
cdef int LL_MAT_PPRINT_ROW_THRESH = 500       # row threshold for choosing print format
cdef int LL_MAT_PPRINT_COL_THRESH = 20        # column threshold for choosing print format

cdef class LLSparseMatrix:
    """
    Linked-List Format matrix.

    Note:
    Despite the name, this matrix doesn't use any linked list.
    """
    ######################################################################################################################
    # Init/Free
    ######################################################################################################################
    cdef:
        public int nrow   # number of rows
        public int ncol   # number of columns
        public int nnz    # number of values stored
        public bint is_symmetric  # true if symmetric matrix

        int     size_hint
        bint    store_zeros
        int     nalloc    # allocated size of value and index arrays
        int     free      # index to first element in free chain
        double *val       # pointer to array of values
        int    *col       # pointer to array of indices
        int    *link      # pointer to array of indices
        int    *root      # pointer to array of indices

    def __cinit__(self, int nrow, int ncol, int size_hint=LL_MAT_DEFAULT_SIZE_HINT, bint store_zeros=False):
        self.nrow = nrow
        self.ncol = ncol
        self.nnz = 0

        self.is_symmetric = False

        if size_hint < 1:
            raise ValueError('size_hint (%d) must be >= 1' % size_hint)

        self.size_hint = size_hint
        self.store_zeros = store_zeros

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
        for i in xrange(nrow):
            root[i] = -1

    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.col)
        PyMem_Free(self.link)
        PyMem_Free(self.root)

    ######################################################################################################################
    # Get/Set items
    ######################################################################################################################
    def _assert_well_formed_indexed_tuple(self, tuple key):
        """
        Assert the tuple given to find an item in the matrix is well formed.

        Args:
          key: A ``tuple`` ``(row, col)``.

        Raises:
          An ``IndexError`` if the ``key`` argument is not well formed.

        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        cdef int i = key[0]
        cdef int j = key[1]

        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            raise IndexError('indices out of range')


    def __setitem__(self, tuple key, double value):
        self._assert_well_formed_indexed_tuple(key)

        cdef int i = key[0]
        cdef int j = key[1]

        if self.is_symmetric and i < j:
          raise IndexError('write operation to upper triangle of symmetric matrix')

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
        self._assert_well_formed_indexed_tuple(key)

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

    ######################################################################################################################
    # Matrix conversions
    ######################################################################################################################
    def to_csr(self):
        pass

    def to_csc(self):
        pass

    ######################################################################################################################
    # String representations
    ######################################################################################################################
    def __repr__(self):
        s = "LLSparseMatrix of size %d by %d with %d values" % (self.nrow, self.ncol, self.nnz)
        return s

    def print_to(self, OUT):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream: anything that supports
        """
        # TODO: adapt to any numbers... and allow for additional parameters to contral the output
        cdef int i, k, first = 1;
        symmetric_str = None

        if self.is_symmetric:
            symmetric_str = 'symmetric'
        else:
            symmetric_str = 'general'

        cdef double *mat
        cdef int j
        cdef double val

        if self.nrow <= LL_MAT_PPRINT_COL_THRESH and self.ncol <= LL_MAT_PPRINT_ROW_THRESH:


            # create linear vector presentation
            # TODO: put in a method of its own
            mat = <double *> PyMem_Malloc(self.nrow * self.ncol * sizeof(double))

            if not mat:
                raise MemoryError()

            for i in xrange(self.nrow):
                for j in xrange(self.ncol):
                    mat[i* self.ncol + j] = 0.0
                k = self.root[i]
                while k != -1:
                    mat[(i*self.ncol)+self.col[k]] = self.val[k]
                    k = self.link[k]

            print('LLSparseMatrix (%s, [%d,%d]):' % (symmetric_str, self.nrow, self.ncol), file=OUT)

            for i in xrange(self.nrow):
                for j in xrange(self.ncol):
                    val = mat[(i*self.ncol)+j]
                    #print('%9.*f ' % (6, val), file=OUT, end='')
                    print('{0:9.6f} '.format(val), end='')
                print()

            PyMem_Free(mat)


########################################################################################################################
# Factory methods
########################################################################################################################
def make_ll_sparse_matrix(**kwargs):
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

        for i in xrange(nrow):
            for j in xrange(ncol):
                value = matrix_view[i, j]
                if value != 0.0:
                    ll_mat[i, j] = value

        return ll_mat
