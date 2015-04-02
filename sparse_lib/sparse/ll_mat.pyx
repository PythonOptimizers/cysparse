from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


cdef class LLSparseMatrix:

  cdef:
    public int nrow        # number of rows
    public int ncol        # number of columns
    public int nnz         # number of values stored
    public int symmetric   # true if symmetric matrix

    int     size_hint
    int     store_zeros    # whether to store zero values
    int     nalloc         # allocated size of value and index arrays
    int     free           # index to first element in free chain
    double *val            # pointer to array of values
    int    *col            # pointer to array of indices
    int    *link           # pointer to array of indices
    int    *root           # pointer to array of indices

  def __cinit__(self, int nrow, int ncol, int size_hint, int store_zeros=0):
    self.nrow = nrow
    self.ncol = ncol
    self.nnz = 0
    self.size_hint = max(1, size_hint)
    self.store_zeros = store_zeros

    val = <double *>PyMem_Malloc(self.size_hint * sizeof(double))
    if not val:
      raise MemoryError()
    self.val = val

    col = <int *>PyMem_Malloc(self.size_hint * sizeof(int))
    if not col:
      raise MemoryError()
    self.col = col

    link = <int *>PyMem_Malloc(self.size_hint * sizeof(int))
    if not link:
      raise MemoryError()
    self.link = link

    root = <int *>PyMem_Malloc(self.nrow * sizeof(int))
    if not root:
      raise MemoryError()
    self.root = root

    cdef int i
    for i in xrange(nrow):
      root[i] = -1


  def __dealloc__(self):
    PyMem_Free(self.val)
    PyMem_Free(self.col)
    PyMem_Free(self.link)
    PyMem_Free(self.root)


  def __repr__(self):
    s = "CySparseMatrix of size %d by %d with %d values" % (self.nrow, self.ncol, self.nnz)
    return s


  def __setitem__(self, tuple key, double value):
    cdef int row = key[0]
    cdef int col = key[1]


  def __getitem__(self, tuple key):
    print "Requesting element at (%d,%d)" % (key[0], key[1])
    return 0.0

