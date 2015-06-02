from __future__ import print_function

########################################################################################################################
# CySparse cimport/import
########################################################################################################################
from cysparse.types.cysparse_types cimport *
from cysparse.types.cysparse_types import type_to_string

from cysparse.sparse.ll_mat cimport LL_MAT_INCREASE_FACTOR

from cysparse.sparse.s_mat cimport unexposed_value
from cysparse.types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type
from cysparse.sparse.ll_mat cimport PyLLSparseMatrix_Check, LL_MAT_PPRINT_COL_THRESH, LL_MAT_PPRINT_ROW_THRESH
from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_INT64_t cimport MutableSparseMatrix_INT32_t_INT64_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT64_t cimport LLSparseMatrix_INT32_t_INT64_t
from cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_INT64_t cimport LLSparseMatrixView_INT32_t_INT64_t



#from cysparse.sparse.csr_mat cimport MakeCSRSparseMatrix, MakeCSRComplexSparseMatrix
#from cysparse.sparse.csc_mat cimport MakeCSCSparseMatrix
#from cysparse.utils.equality cimport values_are_equal
#from cysparse.sparse.IO.mm cimport MakeLLSparseMatrixFromMMFile2, MakeMMFileFromSparseMatrix

from cysparse.sparse.sparse_utils.generate_indices_INT32_t cimport create_c_array_indices_from_python_object_INT32_t

########################################################################################################################
# CySparse include
########################################################################################################################
# pxi files should come last (except for circular dependencies)
include "ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_INT64_t.pxi"
include "ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT64_t.pxi"
include "ll_mat_helpers/ll_mat_multiplication_INT32_t_INT64_t.pxi"


########################################################################################################################
# Cython, NumPy import/cimport
########################################################################################################################
# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

cnp.import_array()

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc,free, calloc
from libc.string cimport memcpy
from cpython cimport PyObject, Py_INCREF

########################################################################################################################
# External code include
########################################################################################################################
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
    PyObject* PyList_GET_ITEM(PyObject *list, Py_ssize_t i)
    int PyList_Check(PyObject *p)
    Py_ssize_t PyList_Size(PyObject *list)

    long PyInt_AS_LONG(PyObject *io)
    PyObject* PyFloat_FromDouble(double v)
    Py_complex PyComplex_AsCComplex(PyObject *op)

cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)

    double creal(double complex z)
    double cimag(double complex z)

    long double creall(long double complex z)
    long double cimagl(long double complex z)

    double cabs(double complex)
    float cabsf(float complex)
    long double cabsl(long double complex)

cdef extern from 'math.h':
    double fabs  (double x)
    float fabsf (float x)
    long double fabsl (long double x)

    double sqrt (double x)
    float sqrtf (float x)
    long double sqrtl (long double x)

cdef extern from "stdlib.h":
    void *memcpy(void *dst, void *src, long n)

########################################################################################################################
# CySparse cimport/import to avoid circular dependencies
########################################################################################################################
from cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_INT64_t cimport LLSparseMatrixView_INT32_t_INT64_t, MakeLLSparseMatrixView_INT32_t_INT64_t


########################################################################################################################
# CLASS LLSparseMatrix
########################################################################################################################
cdef class LLSparseMatrix_INT32_t_INT64_t(MutableSparseMatrix_INT32_t_INT64_t):
    """
    Linked-List Format matrix.

    Note:
        Despite its name, this matrix doesn't use any linked list, only C-arrays.
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

        self.type = "LLSparseMatrix"
        self.type_name = "LLSparseMatrix [INT32_t, INT64_t]"

        # This is particular to the LLSparseMatrix type
        # Do we allocate memory here or
        # do we let another factory method do it for us?
        no_memory = kwargs.get('no_memory', False)

        cdef INT32_t i

        if not no_memory:

            val = <INT64_t *> PyMem_Malloc(self.size_hint * sizeof(INT64_t))
            if not val:
                raise MemoryError()
            self.val = val

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

        temp = <INT64_t *> PyMem_Realloc(self.val, nalloc_new * sizeof(INT64_t))
        if not temp:
            raise MemoryError()
        self.val = <INT64_t *>temp

        self.nalloc = nalloc_new

    cdef _realloc_expand(self):
        """
        Realloc space for internal arrays.

        Note:
            We use ``LL_MAT_INCREASE_FACTOR`` as expanding factor.

        """
        assert LL_MAT_INCREASE_FACTOR > 1.0
        cdef INT32_t real_new_alloc = <INT32_t>(<FLOAT64_t>LL_MAT_INCREASE_FACTOR * self.nalloc) + 1

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
                    self.col[k_new] = self.col[k]
                    self.link[k_new] = self.link[k]
                    k_last = k_new
                else:
                    k_last = k

                k = self.link[k]

        # shrink arrays
        self._realloc(nalloc_new)

        return

    def copy(self):
        """
        Return a (deep) copy of itself.

        Warning:
            Because we use memcpy and thus copy memory internally, we have to be careful
        """
        # Warning: Because we use memcpy and thus copy memory internally, we have to be careful to always update this method
        # whenever the LLSparseMatrix class changes...

        cdef LLSparseMatrix_INT32_t_INT64_t self_copy

        # we copy manually the C-arrays
        self_copy = LLSparseMatrix_INT32_t_INT64_t(control_object=unexposed_value, no_memory=True, nrow=self.nrow, ncol=self.ncol, size_hint=self.size_hint, store_zeros=self.store_zeros, is_symmetric=self.is_symmetric)

        # copy C-arrays
        cdef:
            INT64_t * val
            INT32_t * col
            INT32_t * link
            INT32_t * root

        val = <INT64_t *> PyMem_Malloc(self.nalloc * sizeof(INT64_t))
        if not val:
            raise MemoryError()
        memcpy(val, self.val, self.nalloc * sizeof(INT64_t))
        self_copy.val = val

        col = <INT32_t *> PyMem_Malloc(self.nalloc * sizeof(INT32_t))
        if not col:
            raise MemoryError()
        memcpy(col, self.col, self.nalloc * sizeof(INT32_t))
        self_copy.col = col

        link = <INT32_t *> PyMem_Malloc(self.nalloc * sizeof(INT32_t))
        if not link:
            raise MemoryError()
        memcpy(link, self.link, self.nalloc * sizeof(INT32_t))
        self_copy.link = link

        root = <INT32_t *> PyMem_Malloc(self.nrow * sizeof(INT32_t))
        if not root:
            raise MemoryError()
        memcpy(root, self.root, self.nrow * sizeof(INT32_t))
        self_copy.root = root

        self_copy.nalloc = self.nalloc
        self_copy.free = self.free
        self_copy.nnz = self.nnz

        return self_copy

    def generalize(self):
        """
        Convert matrix from symmetric to non-symmetric form (in-place).
        """
        cdef:
            INT32_t k, i, j

        if self.is_symmetric:

            self.is_symmetric = False  # to allow writing in upper triangle

            for i from 0 <= i < self.nrow:
                k = self.root[i]
                while k != -1:
                    j = self.col[k]

                    if i > j:
                        self.put(j, i, self.val[k])

                    k = self.link[k]

    def memory_real(self):
        """
        Return the real amount of memory used internally for the matrix.

        Returns:
            The exact number of bits used to store the matrix (but not the object in itself, only the internal memory
            needed to store the matrix).

        """
        cdef INT32_t total_memory = 0

        # root
        total_memory += self.nrow * sizeof(INT32_t)
        # col
        total_memory += self.nalloc * sizeof(INT32_t)
        # link
        total_memory += self.nalloc * sizeof(INT32_t)
        # val
        total_memory += self.nalloc * sizeof(INT64_t)

        return total_memory

    ####################################################################################################################
    # SORTING
    ####################################################################################################################
    cdef bint is_sorted(self):
        """
        Tell if matrix is sorted, i.e. if its column indices are sorted row by row as it is supposed to be.
        """
        cdef INT32_t k, i, last_index

        for i from 0 <= i < self.nrow:
            # column index of first element in row i
            k = self.root[i]
            if k != -1:
                last_index = self.col[k]
                k = self.link[k]

            # compare column indices
            while k != -1:
                if self.col[k] <= last_index:
                    return False  # column indices are not sorted!

                last_index = self.col[k]
                k = self.link[k]

        return True  # not bad column index detected

    ####################################################################################################################
    # SUB-MATRICES
    ####################################################################################################################
    ####################################################################################################################
    #                                            ### CREATE ###
    # TODO: to be done
    cdef create_submatrix(self, PyObject* obj1, PyObject* obj2):
        raise NotImplementedError("Not implemented yet...")
        cdef:
            INT32_t nrow
            INT32_t * row_indices,
            INT32_t ncol
            INT32_t * col_indices
            INT32_t i, j

        row_indices = create_c_array_indices_from_python_object_INT32_t(self.nrow, obj1, &nrow)
        col_indices = create_c_array_indices_from_python_object_INT32_t(self.ncol, obj2, &ncol)

    ####################################################################################################################
    #                                            ### ASSIGN ###
    cdef assign(self, LLSparseMatrixView_INT32_t_INT64_t view, object obj):
        """
        Set ``A[..., ...] = obj`` directly.

        Args:
            view: An ``LLSparseMatrixView_INT32_t_INT64_t`` that points to this matrix (``self``).
            obj: Any Python object that implements ``__getitem__()`` and accepts a ``tuple`` ``(i, j)``.

        Note:
            This assignment is done as if ``A[i, j] = val`` was done explicitely. In particular if ``store_zeros``
            is ``True`` and ``obj`` contains zeros, they will be explicitely added.

        Warning:
            There are not test whatsoever.
        """
        # test if view correspond...
        assert self == view.A, "LLSparseMatrixView should correspond to LLSparseMatrix!"

        # TODO: refine this method. It is too generic to do any optimization at all...

        # VIEW
        cdef:
            INT32_t * row_indices = view.row_indices
            INT32_t nrow = view.nrow
            INT32_t * col_indices = view.col_indices
            INT32_t ncol = view.ncol

        cdef:
            INT32_t i, j

        if self.is_symmetric:
            if PyLLSparseMatrix_Check(obj):
                # obj is LLSparseMatrix
                for i from 0 <= i < nrow:
                    for j from 0 <= j <= i:
                        self.put(row_indices[i], col_indices[j], obj.at(i, j))

            else:
                for i from 0 <= i < nrow:
                    for j from 0 <= j <= i:
                        self.put(row_indices[i], col_indices[j], <INT64_t> obj[tuple(i, j)])

        else:   # self.is_symmetric == False

            if PyLLSparseMatrix_Check(obj):
                # obj is LLSparseMatrix
                for i from 0 <= i < nrow:
                    for j from 0 <= j < ncol:
                        self.put(row_indices[i], col_indices[j], obj.at(i, j))

            else:
                for i from 0 <= i < nrow:
                    for j from 0 <= j < ncol:
                        self.put(row_indices[i], col_indices[j], <INT64_t> obj[tuple(i, j)])

    ####################################################################################################################
    # COUNTING ELEMENTS
    ####################################################################################################################
    # TODO: to be done
    cdef count_nnz_from_indices(self, INT32_t * row_indices,INT32_t row_indices_length, INT32_t * col_indices, INT32_t col_indices_length):
        """
        Counts the nnz specified by row and column indices.

        Note:
            A row or column index can be repeated and indices are **not** supposed to be sorted.

        Warning:
            This method is costly, use with care.
        """
        raise NotImplementedError("Not implemented yet...")

    ####################################################################################################################
    # Set/Get individual elements
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    cdef put(self, INT32_t i, INT32_t j, INT64_t value):
        """
        Set :math:`A[i, j] = \textrm{value}` directly.

        Note:
            Store zero elements **only** if ``store_zeros`` is ``True``.

        Warning:
            No out of bound check.

        See:
            :meth:`safe_put`.


        """
        if self.is_symmetric and i < j:
            raise IndexError('Write operation to upper triangle of symmetric matrix not allowed')

        cdef INT32_t k, new_elem, last, col

        # Find element to be set (or removed)
        col = last = -1
        k = self.root[i]
        while k != -1:
            col = self.col[k]
            # See maintenance doc
            if col >= j:
                break
            last = k
            k = self.link[k]

        # Store value
        if self.store_zeros or value != 0.0:
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
            # value == 0.0:
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


    cdef int safe_put(self, INT32_t i, INT32_t j, INT64_t value)  except -1:
        """
        Set ``A[i, j] = value`` directly.

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

        Note:
            We suppose that the elements of a row are **ordered** by ascending column indices.

        See:
            :meth:`safe_at`.


        """
        cdef INT32_t k, t

        if self.is_symmetric and i < j:
            t = i; i = j; j = t

        k = self.root[i]

        while k != -1:
            # version **whitout** order
            #if self.col[k] == j:
            #    return self.val[k]
            #k = self.link[k]

            # version **with** order
            if self.col[k] >= j:
                if self.col[k] == j:
                    return self.val[k]
                break

            k = self.link[k]

        # TODO: test if this return is casted like it should, especially for complex numbers...
        return 0

    # EXPLICIT TYPE TESTS

    cdef INT64_t safe_at(self, INT32_t i, INT32_t j) except? 1:

        """
        Return element ``(i, j)`` but with check for out of bounds indices.

        Raises:
            IndexError: when index out of bound.


        """
        if not 0 <= i < self.nrow or not 0 <= j < self.ncol:
            raise IndexError("Index out of bounds")

        return self.at(i, j)

    ####################################################################################################################
    # __setitem/__getitem__
    ####################################################################################################################
    def __setitem__(self, tuple key, value):
        """
        A[i, j] = value

        Raises:
            IndexError: when index out of bound.


        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        #cdef LLSparseMatrixView view

        ## test for direct access (i.e. both elements are integers)
        #if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
        #    # TODO: don't create temp object
        #    view = MakeLLSparseMatrixView(self, <PyObject *>key[0], <PyObject *>key[1])
        #    self.assign(view, value)

        #    del view
        #    return

        cdef INT32_t i = key[0]
        cdef INT32_t j = key[1]

        self.safe_put(i, j, value)

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

        cdef LLSparseMatrixView_INT32_t_INT64_t view

        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            view =  MakeLLSparseMatrixView_INT32_t_INT64_t(self, <PyObject *>key[0], <PyObject *>key[1])
            return view

        cdef INT32_t i = key[0]
        cdef INT32_t j = key[1]

        return self.safe_at(i, j)

    ####################################################################################################################
    # Set/Get list of elements
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    def put_triplet(self, list index_i, list index_j, list val):
        """
        Assign triplet :math:`\{(i, j, \textrm{val})\}` values to the matrix..


        """
        # TODO: to be completely rewritten
        cdef Py_ssize_t index_i_length = len(index_i)
        cdef Py_ssize_t index_j_length = len(index_j)
        cdef Py_ssize_t val_length = len(val)

        assert index_j_length == index_j_length == val_length, "All lists must be of equal length"

        cdef Py_ssize_t i
        cdef PyObject * elem

        for i from 0 <= i < index_i_length:
            self.safe_put(index_i[i], index_j[i], val[i])

    ####################################################################################################################
    #                                            *** GET ***
    cpdef take_triplet(self, id1, id2, cnp.ndarray[cnp.npy_int64, ndim=1] b):
        """
        Grab values and populate b with it.

        This operation is equivalent to

            for i in range(len(b)):
                b[i] = A[id1[i],id2[i]]

        Args:
            id1, id2: List or :program:`NumPy` arrays with indices. Both **must** be of the same type. In case of :program:`NumPy` arrays, they must
                contain elements of type INT32_t.
            b: :program:`NumpY` array to fill with the values.

        Raises:
            ``TypeError`` is both arguments to give indices are not of the same type (``list`` or :program:`NumPy` arrays) or if
            one of the argument is not a ``list`` or a :program:`NumPy` array.

            ``IndexError`` whenever length don't match.

            A supplementary condition holds when :program:`NumPy` arrays are used to give the indices:

            - the indices arrays **must** be C-contiguous and
            - index elements **must** be of same type than the ``itype`` of the matrix.

            In both cases, a ``TypeError`` is raised.

        Note:
            This method is not as rich as its :program:`PySparse` equivalent but at the same time accept ``list``\s for the indices.

        """
        # TODO: test, test, test!!!
        cdef:
            Py_ssize_t id1_list_length, id2_list_length, i_list # in case we have lists
            INT32_t id1_array_length, id2_array_length, i_array  # in case we have numpy arrays

        # direct access to NumPy vector b
        cdef INT64_t * b_data

        # if indices arrays are given by NumPy arrays
        cdef INT32_t * id1_data
        cdef INT32_t * id2_data

        # stride size if any
        cdef size_t sd = sizeof(INT64_t)
        cdef INT32_t incx = b.strides[0] / sd

        # test arguments
        if PyList_Check(<PyObject *>id1) and PyList_Check(<PyObject *>id2):
            id1_list_length = PyList_Size(<PyObject *>id1)
            id2_list_length = PyList_Size(<PyObject *>id2)
            if id1_list_length != id2_list_length:
                raise IndexError('Both indices lists must be of same size')

            if b.size != id1_list_length:
                raise IndexError('NumPy array must be of the same size than the indices lists')

            # direct access to vector b
            b_data = <INT64_t *> cnp.PyArray_DATA(b)

            if cnp.PyArray_ISCONTIGUOUS(b):
                # fill vector
                for i_list from 0 <= i_list < id1_list_length:
                    b_data[i_list] = self.safe_at(PyInt_AS_LONG(PyList_GET_ITEM(<PyObject *>id1, i_list)), PyInt_AS_LONG(PyList_GET_ITEM(<PyObject *>id2, i_list)))
            else:  # non contiguous array
                # fill vector
                for i_list from 0 <= i_list < id1_list_length:
                    b_data[i_list*incx] = self.safe_at(PyInt_AS_LONG(PyList_GET_ITEM(<PyObject *>id1, i_list)), PyInt_AS_LONG(PyList_GET_ITEM(<PyObject *>id2, i_list)))

        elif cnp.PyArray_Check(id1) and cnp.PyArray_Check(id2):
            id1_array_length = id1.size
            id2_array_length = id2.size
            if id1_array_length != id2_array_length:
                raise IndexError('Both indices lists must be of same size')

            if b.size != id1_array_length:
                raise IndexError('NumPy array must be of the same size than the indices lists')

            if not cnp.PyArray_ISCONTIGUOUS(id1) or not cnp.PyArray_ISCONTIGUOUS(id2):
                raise TypeError('Both NumPy indices arrays must be C-contiguous')

            if not are_mixed_types_compatible(INT32_T, id1.dtype) or not are_mixed_types_compatible(INT32_T, id2.dtype):
                raise TypeError('Both NumPy indices arrays must contain elements of the right index type (%s)' % cysparse_to_numpy_type(INT32_T))

            # direct access to vector b
            b_data = <INT64_t *> cnp.PyArray_DATA(b)

            # direct access to indices arrays
            id1_data = <INT32_t *> cnp.PyArray_DATA(id1)
            id2_data = <INT32_t *> cnp.PyArray_DATA(id2)

            if cnp.PyArray_ISCONTIGUOUS(b):
                # fill vector
                for i_array from 0 <= i_array < id1_array_length:
                    b_data[i_array] = self.safe_at(id1_data[i_array], id2_data[i_array])
            else:  # non contiguous array
                # fill vector
                for i_array from 0 <= i_array < id1_array_length:
                    b_data[i_list*incx] = self.safe_at(id1_data[i_array], id2_data[i_array])

        else:
            raise TypeError('Both arguments with indices must be of the same type (list or NumPy arrays)')

    cpdef object keys(self):
        """
        Return a list of tuples (i,j) of non-zero matrix entries.


        """
        cdef:
            #list list_container
            PyObject *list_p # the list that will hold the keys
            INT32_t i, j, k
            Py_ssize_t pos = 0    # position in list

        if not self.is_symmetric:

            # create list
            list_p = PyList_New(self.nnz)
            if list_p == NULL:
                raise MemoryError()

            for i from 0 <= i < self.nrow:
                k = self.root[i]
                while k != -1:
                    j = self.col[k]
                    # only valid because we use a **new** list, see C API
                    PyList_SET_ITEM(list_p, pos, Py_BuildValue("ii", i, j))
                    pos += 1
                    k = self.link[k]
        else:
            raise NotImplementedError("keys() is not (yet) implemented for symmetrical LLSparseMatrix")

        return <object> list_p

    cpdef object values(self):
        """
        Return a list of the non-zero matrix entries.
        """
        cdef:
            PyObject *list_p   # the list that will hold the values
            INT32_t i, k
            Py_ssize_t pos = 0        # position in list

        if not self.is_symmetric:
            list_p = PyList_New(self.nnz)
            if list_p == NULL:
                raise MemoryError()

            # EXPLICIT TYPE TESTS

            for i from 0<= i < self.nrow:
                k = self.root[i]
                while k != -1:

                    PyList_SET_ITEM(list_p, pos, Py_BuildValue("l", self.val[k]))


                    pos += 1
                    k = self.link[k]

        else:
            raise NotImplementedError("values() not (yet) implemented for symmetrical LLSparseMatrix")

        return <object> list_p


    cpdef object items(self):
        """
        Return a list of tuples (indices, value) of the non-zero matrix entries' keys and values.

        The indices are themselves tuples (i,j) of row and column values.

        """
        cdef:
            PyObject *list_p;     # the list that will hold the values
            INT32_t i, j, k
            Py_ssize_t pos = 0         # position in list
            INT64_t val

        list_p = PyList_New(self.nnz)
        if list_p == NULL:
            raise MemoryError()

        # EXPLICIT TYPE TESTS

        for i from 0 <= i < self.nrow:
            k = self.root[i]
            while k != -1:
                j = self.col[k]
                val = self.val[k]

                PyList_SET_ITEM(list_p, pos, Py_BuildValue("((ii)l)", i, j, self.val[k]))


                pos += 1

                k = self.link[k]

        return <object> list_p

    cpdef find(self):
        """
        Return 3 NumPy arrays with the non-zero matrix entries: i-rows, j-cols, vals.
        """
        cdef cnp.npy_intp dmat[1]
        dmat[0] = <cnp.npy_intp> self.nnz

        # EXPLICIT TYPE TESTS

        cdef:


            cnp.ndarray[cnp.int32_t, ndim=1] a_row = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)
            cnp.ndarray[cnp.int32_t, ndim=1] a_col = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)



            cnp.ndarray[cnp.int64_t, ndim=1] a_val = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT64)


            INT32_t   *pi, *pj;   # Intermediate pointers to matrix data
            INT64_t    *pv;
            INT32_t   i, k, elem;

        pi = <INT32_t *> cnp.PyArray_DATA(a_row)
        pj = <INT32_t *> cnp.PyArray_DATA(a_col)
        pv = <INT64_t *> cnp.PyArray_DATA(a_val)

        elem = 0
        for i from 0 <= i < self.nrow:
            k = self.root[i]
            while k != -1:
                pi[ elem ] = i
                pj[ elem ] = self.col[k]
                pv[ elem ] = self.val[k]
                k = self.link[k]
                elem += 1

        return (a_row, a_col, a_val)

    ####################################################################################################################
    # Addition
    ####################################################################################################################
    def shift(self, sigma, LLSparseMatrix_INT32_t_INT64_t B):

        if self.nrow != B.nrow or self.ncol != B.ncol:
            raise IndexError('Matrix shapes do not match')

        if not is_scalar(sigma):
            raise TypeError('sigma must be a scalar')

        cdef:
            INT64_t casted_sigma, v
            INT32_t k, i, j

        try:
            casted_sigma = <INT64_t> sigma
        except:
            raise TypeError('Factor sigma is not compatible with the dtype (%d) of this matrix' % type_to_string(self.dtype))

        if self.is_symmetric == B.is_symmetric:
            # both matrices are symmetric or are not symmetric
            for i from 0 <= i < B.nrow:
                k = B.root[i]

                while k != -1:
                    update_ll_mat_item_add_INT32_t_INT64_t(self, i, B.col[k], casted_sigma * B.val[k])
                    k = B.link[k]

        elif B.is_symmetric:
            # self is not symmetric
            for i from 0 <= i < B.nrow:
                k = B.root[i]

                while k != -1:
                    j = B.col[k]
                    v = casted_sigma * B.val[k]
                    update_ll_mat_item_add_INT32_t_INT64_t(self, i, j, v)
                    if i != j:
                        update_ll_mat_item_add_INT32_t_INT64_t(self, j, i, v)
                    k = B.link[k]
        else:
            # B is not symmetric but self is symmetric
            # doesn't make sense...
            raise TypeError('Cannot shift symmetric matrix by non-symmetric matrix')




    ####################################################################################################################
    # Multiplication
    ####################################################################################################################
    def matvec(self, B):
        """
        Return :math:`A * b`.
        """
        return multiply_ll_mat_with_numpy_vector_INT32_t_INT64_t(self, B)

    def matvec_transp(self, B):
        """
        Return :math:`A^t * b`.
        """
        return multiply_transposed_ll_mat_with_numpy_vector_INT32_t_INT64_t(self, B)

    def __mul__(self, B):
        """
        Classical matrix multiplication.

        Cases:

        - ``C = A * B`` where `B` is an ``LLSparseMatrix`` matrix. ``C`` is an ``LLSparseMatrix`` of same type.
        - ``C = A * B`` where ``B`` is an :program:`NumPy` matrix. ``C`` is a dense :program:`NumPy` matrix. (not yet implemented).
        """
        # CASES
        if PyLLSparseMatrix_Check(B):
            return multiply_two_ll_mat_INT32_t_INT64_t(self, B)
            #raise NotImplementedError("Multiplication with this kind of object not implemented yet...")
        elif cnp.PyArray_Check(B):
            # test type
            assert are_mixed_types_compatible(INT64_T, B.dtype), "Multiplication only allowed with a Numpy compatible type (%s)!" % cysparse_to_numpy_type(INT64_T)

            if B.ndim == 2:
                #return multiply_ll_mat_with_numpy_ndarray(self, B)
                raise NotImplementedError("Multiplication with this kind of object not implemented yet...")
            elif B.ndim == 1:
                return self.matvec(B)
            else:
                raise IndexError("Matrix dimensions must agree")
        else:
            raise NotImplementedError("Multiplication with this kind of object not implemented yet...")

    #def __rmul__(self, B):

    def __imul__(self, B):
        """
        Classical in place multiplication ``A *= B``.
        """
        # TODO: add tests and error messages
        self.scale(B)
        return self

    ####################################################################################################################
    # Scaling
    ####################################################################################################################
    def scale(self, INT64_t sigma):
        """
        Scale each element in the matrix by the constant ``sigma``.

        Args:
            sigma:
        """
        cdef:
            INT32_t k, i

        for i from 0 <= i < self.nrow:
            k = self.root[i]

            while k != -1:

                self.val[k] *= sigma

                k = self.link[k]

    def col_scale(self, cnp.ndarray[cnp.npy_int64, ndim=1] v):
        """
        Scale the i:sup:`th` column of A by ``v[i]`` in place for ``i=0, ..., ncol-1``

        Args:
            v:
        """
        # TODO: maybe accept something else than only an numpy array?
        # TODO: benchmark.. we don't use the same approach than PySparse at all

        # test dimensions
        if self.ncol != v.size:
            raise IndexError("Dimensions must agree ([%d,%d] and [%d, %d])" % (self.nrow, self.ncol, v.size, 1))

        cdef:
            INT32_t k, i

        # strides
        cdef size_t sd = sizeof(INT64_t)
        cdef INT32_t incx

        # direct access to vector v
        cdef INT64_t * v_data = <INT64_t *> cnp.PyArray_DATA(v)

        # test if v vector is C-contiguous or not
        if cnp.PyArray_ISCONTIGUOUS(v):
            for i from 0 <= i < self.nrow:
                k = self.root[i]
                while k != -1:
                    self.val[k] *= v_data[self.col[k]]

                    k = self.link[k]

        else:
            incx = v.strides[0] / sd
            for i from 0 <= i < self.nrow:
                k = self.root[i]
                while k != -1:
                    self.val[k] *= v_data[self.col[k]*incx]

                    k = self.link[k]

    def row_scale(self, cnp.ndarray[cnp.npy_int64, ndim=1] v):
        """
        Scale the i:sup:`th` row of A by ``v[i]`` in place for ``i=0, ..., nrow-1``

        Args:
            v:
        """
        # TODO: maybe accept something else than only an numpy array?

        # test dimensions
        if self.nrow != v.size:
            raise IndexError("Dimensions must agree ([%d,%d] and [%d, %d])" % (self.nrow, self.ncol, v.size, 1))

        cdef:
            INT32_t k, i
            INT64_t val

        # strides
        cdef size_t sd = sizeof(INT64_t)
        cdef INT32_t incx

        # direct access to vector v
        # TODO: it could be worth to copy the array in case of stride...
        cdef INT64_t * v_data = <INT64_t *> cnp.PyArray_DATA(v)

        # test if v vector is C-contiguous or not
        if cnp.PyArray_ISCONTIGUOUS(v):

            for i from 0 <= i < self.nrow:
                k = self.root[i]
                val = v_data[i]

                while k != -1:
                    self.val[k] *= val

                    k = self.link[k]

        else:
            incx = v.strides[0] / sd
            for i from 0 <= i < self.nrow:
                k = self.root[i]
                val = v_data[i * incx]

                while k != -1:
                    self.val[k] *= val

                    k = self.link[k]

    ####################################################################################################################
    # Norms
    ####################################################################################################################
    def norm(self, norm_name):
        """
        Computes a norm of the matrix.

        Args:
            norm_name: Can be '1', 'inf' or 'frob'.

        Note:
            **All** norms have been thoroughly tested for ``dtype == FLOAT64_T`` and ``itype == INT32_T``.
        """
        if norm_name == 'inf': # ||A||_\infty
            return self._norm_inf()
        elif norm_name == '1': # ||A||_1
            return self._norm_one()
        elif norm_name == 'frob': # Frobenius norm
            return self._norm_frob()
        else:
            raise NotImplementedError("This type ('%s') of norm is not implemented (yet?)" % norm_name)

    cdef _norm_one(self):
        """
        Computes :math:`||A||_1`.

        Warning:
            Only works if the matrix is **not** symmetric!

        """
        cdef:
            FLOAT64_t max_col_sum
            INT32_t i, k

        # create temp array for column results
        cdef FLOAT64_t * col_sum = <FLOAT64_t *> calloc(self.ncol, sizeof(FLOAT64_t))
        if not col_sum:
            raise MemoryError()

        if self.is_symmetric:

            # compute sum of columns
            for i from 0<= i < self.nrow:
                k = self.root[i]

                # EXPLICIT TYPE TESTS
                while k != -1:

                    if self.col[k] != i:
                        col_sum[self.col[k]] +=  fabs(<FLOAT64_t>self.val[k])
                    col_sum[i] +=  fabs(<FLOAT64_t>self.val[k])

                    k = self.link[k]

        else:  # not symmetric

            # compute sum of columns
            for i from 0<= i < self.nrow:
                k = self.root[i]

                # EXPLICIT TYPE TESTS
                while k != -1:

                    col_sum[self.col[k]] += fabs(<FLOAT64_t>self.val[k])

                    k = self.link[k]

        # compute max of all column sums
        max_col_sum = <FLOAT64_t> 0.0

        for i from 0 <= i < self.ncol:
            if col_sum[i] > max_col_sum:
                max_col_sum = col_sum[i]

        free(col_sum)

        return max_col_sum

    cdef _norm_inf(self):
        """
        Computes :math:`||A||_\infty`.

        """
        cdef:
            FLOAT64_t max_row_sum, row_sum
            INT32_t i, k

        # for symmetric case
        cdef FLOAT64_t * row_sum_array

        max_row_sum = <FLOAT64_t> 0.0

        if not self.is_symmetric:
            for i from 0<= i < self.nrow:
                k = self.root[i]

                row_sum = <FLOAT64_t> 0.0

                # EXPLICIT TYPE TESTS
                while k != -1:

                    row_sum += fabs(<FLOAT64_t>self.val[k])

                    k = self.link[k]

                if row_sum > max_row_sum:
                    max_row_sum = row_sum

        else:  # matrix is symmetric

            # create temp array for column results
            row_sum_array = <FLOAT64_t *> calloc(self.nrow, sizeof(FLOAT64_t))

            if not row_sum_array:
                raise MemoryError()

            for i from 0<= i < self.nrow:
                k = self.root[i]

                # EXPLICIT TYPE TESTS
                while k != -1:

                    if self.col[k] != i:
                        row_sum_array[self.col[k]] += fabs(<FLOAT64_t>self.val[k])
                    row_sum_array[i] += fabs(<FLOAT64_t>self.val[k])

                    k = self.link[k]

            # compute max of all row sums
            for i from 0 <= i < self.nrow:
                if row_sum_array[i] > max_row_sum:
                    max_row_sum = row_sum_array[i]

            free(row_sum_array)

        return max_row_sum

    cdef _norm_frob(self):
        """
        Computes the Frobenius norm.


        """
        cdef:
            FLOAT64_t norm_sum, norm
            INT32_t i, k
            FLOAT64_t abs_val, abs_val_square

        norm_sum = <FLOAT64_t> 0.0

        for i from 0<= i < self.nrow:
            k = self.root[i]

            # EXPLICIT TYPE TESTS
            while k != -1:

                abs_val = fabs(<FLOAT64_t> self.val[k])


                abs_val_square = abs_val * abs_val
                norm_sum += abs_val_square
                if self.is_symmetric and i != self.col[k]:
                    norm_sum += abs_val_square

                k = self.link[k]


        norm = sqrt(norm_sum)


        return norm


    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def print_to(self, OUT, width=9, print_big_matrices=False, transposed=False):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.

        """
        # EXPLICIT TYPE TESTS
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        # TODO: don't create temporary matrix
        cdef INT32_t i, k, first = 1

        print(self._matrix_description_before_printing(), file=OUT)

        cdef INT64_t *mat
        cdef INT32_t j
        cdef INT64_t val, ival

        if not self.nnz:
            return

        if print_big_matrices or (self.nrow <= LL_MAT_PPRINT_COL_THRESH and self.ncol <= LL_MAT_PPRINT_ROW_THRESH):
            # create linear vector presentation

            mat = <INT64_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(INT64_t))

            if not mat:
                raise MemoryError()

            # CREATION OF TEMP MATRIX
            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:

                    mat[i* self.ncol + j] = 0

                k = self.root[i]
                while k != -1:
                    mat[(i*self.ncol)+self.col[k]] = self.val[k]
                    if self.is_symmetric:
                        mat[(self.col[k]*self.ncol)+i] = self.val[k]
                    k = self.link[k]

            # PRINTING OF TEMP MATRIX
            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    val = mat[(i*self.ncol)+j]

                    print('{:{width}.6f} '.format(val, width=width), end='', file=OUT)

                print(file=OUT)

            PyMem_Free(mat)

        else:
            print('Matrix too big to print out', file=OUT)