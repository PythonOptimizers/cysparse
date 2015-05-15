from __future__ import print_function

from cysparse.types.cysparse_types cimport *
from cysparse.sparse.ll_mat cimport LL_MAT_INCREASE_FACTOR

from cysparse.sparse.sparse_mat cimport unexposed_value
from cysparse.sparse.sparse_mat_matrices.sparse_mat_INT32_t_FLOAT32_t cimport MutableSparseMatrix_INT32_t_FLOAT32_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT32_t cimport LLSparseMatrix_INT32_t_FLOAT32_t
#from cysparse.sparse.csr_mat cimport MakeCSRSparseMatrix, MakeCSRComplexSparseMatrix
#from cysparse.sparse.csc_mat cimport MakeCSCSparseMatrix
#from cysparse.utils.equality cimport values_are_equal
#from cysparse.sparse.IO.mm cimport MakeLLSparseMatrixFromMMFile2, MakeMMFileFromSparseMatrix


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

cdef extern from "Python.h":
    PyObject* Py_BuildValue(char *format, ...)
    PyObject* PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(PyObject *list, Py_ssize_t i, PyObject *o)
    PyObject* PyFloat_FromDouble(double v)


cdef class LLSparseMatrix_INT32_t_FLOAT32_t(MutableSparseMatrix_INT32_t_FLOAT32_t):
    """
    Linked-List Format matrix.

    Note:
        Despite its name, this matrix doesn't use any linked list.
    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    def __cinit__(self,  **kwargs):
        """

        Args:
            no_memory: ``False`` by default. If ``True``, the constructor doesn't allocate any memory, otherwise,
            memory is preallocated for the internal C arrays. When no memory is allocated, a factory method **must** provide it!
        """
        if self.size_hint < 1:
            raise ValueError('size_hint (%d) must be >= 1' % self.size_hint)

        self.type_name = "LLSparseMatrix [INT32_t, FLOAT32_t]"

        # This is particular to the LLSparseMatrix type
        # Do we allocate memory here or
        # do we let another factory method do it for us?
        no_memory = kwargs.get('no_memory', False)

        cdef INT32_t i

        if not no_memory:

            val = <FLOAT32_t *> PyMem_Malloc(self.size_hint * sizeof(FLOAT32_t))
            if not val:
                raise MemoryError()
            self.val = val

            if self.is_complex:
                ival = <FLOAT32_t *> PyMem_Malloc(self.size_hint * sizeof(FLOAT32_t))
                if not ival:
                    raise MemoryError()
                self.ival = ival

            col = <INT32_t *> PyMem_Malloc(self.size_hint * sizeof(INT32_t))
            if not col:
                raise MemoryError()
            self.col = col

            link = <INT32_t *> PyMem_Malloc(self.size_hint * sizeof(INT32_t))
            if not link:
                raise MemoryError()
            self.link = link

            root = <INT32_t *> PyMem_Malloc(self.nrow * sizeof(INT32_t))
            if not root:
                raise MemoryError()
            self.root = root

            self.nalloc = self.size_hint
            self.free = -1

            for i from 0 <= i < self.nrow:
                root[i] = -1

    def __dealloc__(self):
        """
        """
        PyMem_Free(self.val)
        if self.is_complex:
            PyMem_Free(self.ival)
        PyMem_Free(self.col)
        PyMem_Free(self.link)
        PyMem_Free(self.root)

    cdef _realloc(self, INT32_t nalloc_new):
        """
        Realloc space for the internal arrays.

        Note:
            Internal arrays can be expanded or shrunk.

        """
        cdef:
            void *temp

        temp = <INT32_t *> PyMem_Realloc(self.col, nalloc_new * sizeof(INT32_t))
        if not temp:
            raise MemoryError()
        self.col = <INT32_t*>temp

        temp = <INT32_t *> PyMem_Realloc(self.link, nalloc_new * sizeof(INT32_t))
        if not temp:
            raise MemoryError()
        self.link = <INT32_t *>temp

        temp = <FLOAT32_t *> PyMem_Realloc(self.val, nalloc_new * sizeof(FLOAT32_t))
        if not temp:
            raise MemoryError()
        self.val = <FLOAT32_t *>temp

        if self.is_complex:
            temp = <FLOAT32_t *> PyMem_Realloc(self.ival, nalloc_new * sizeof(FLOAT32_t))
            if not temp:
                raise MemoryError()
            self.ival = <FLOAT32_t *>temp

        self.nalloc = nalloc_new

    cdef _realloc_expand(self):
        """
        Realloc space for internal arrays.

        Note:
            We use ``LL_MAT_INCREASE_FACTOR`` as expanding factor.

        
        
        """
        assert LL_MAT_INCREASE_FACTOR > 1.0
        cdef INT32_t real_new_alloc = <INT32_t>(<FLOAT32_t>LL_MAT_INCREASE_FACTOR * self.nalloc) + 1

        return self._realloc(real_new_alloc)

    def compress(self):
        """
        Shrink matrix to its minimal size.

        """
        cdef:
            INT32_t i, k, k_next, k_last, k_new, nalloc_new;

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

        """
        cdef INT32_t total_memory = 0

        print( "alleluia: " + str(self.nalloc))
        # root
        total_memory += self.nrow * sizeof(INT32_t)
        # col
        total_memory += self.nalloc * sizeof(INT32_t)
        # link
        total_memory += self.nalloc * sizeof(INT32_t)
        # val
        total_memory += self.nalloc * sizeof(FLOAT32_t)

        if self.is_complex:
            # ival
            total_memory += self.nalloc * sizeof(FLOAT32_t)

        return total_memory

#cdef NewLLSparseMatrix_INT32_t_FLOAT32_t(FLOAT32_t * val, INT32_t *root, INT32_t *link, INT32_t *col, FLOAT32_t * ival):
#        pass