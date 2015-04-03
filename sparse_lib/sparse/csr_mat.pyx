"""
csr_mat extension.


"""

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class CSRSparseMatrix:
  """
  Compressed Sparse Row Format matrix.

  Note:
    This matrix can **not** be modified.

  """
  cdef:
    public int nrow  # number of rows
    public int ncol  # number of columns
    public int nnz   # number of values stored
    public int symmetric  # true if symmetric matrix

    int     size_hint
    int     store_zeros  # whether to store zero values
    int     nalloc       # allocated size of value and index arrays


    double *val;		 # pointer to array of values
    int    *col;		 # pointer to array of indices
    int    *ind;		 # pointer to array of indices

  def __cinit__(self, int nrow, int ncol, int size_hint, int store_zeros=0):
    self.nrow = nrow
    self.ncol = ncol
    self.nnz = 0
    self.size_hint = max(1, size_hint)
    self.store_zeros = store_zeros

    val = <double *> PyMem_Malloc(self.size_hint * sizeof(double))
    if not val:
      raise MemoryError()
    self.val = val

    col = <int *> PyMem_Malloc(self.size_hint * sizeof(int))
    if not col:
      raise MemoryError()
    self.col = col

    ind = <int *> PyMem_Malloc(self.size_hint * sizeof(int))
    if not ind:
      raise MemoryError()
    self.ind = ind

  def __dealloc__(self):
    PyMem_Free(self.val)
    PyMem_Free(self.col)
    PyMem_Free(self.ind)
