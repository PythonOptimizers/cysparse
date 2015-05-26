from __future__ import print_function

########################################################################################################################
# CySparse cimport/import
########################################################################################################################
from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat cimport LL_MAT_INCREASE_FACTOR

from cysparse.sparse.sparse_mat cimport unexposed_value
from cysparse.types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type
from cysparse.sparse.ll_mat cimport PyLLSparseMatrix_Check, LL_MAT_PPRINT_COL_THRESH, LL_MAT_PPRINT_ROW_THRESH
from cysparse.sparse.sparse_mat_matrices.sparse_mat_INT32_t_INT32_t cimport MutableSparseMatrix_INT32_t_INT32_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT32_t cimport LLSparseMatrix_INT32_t_INT32_t
from cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_INT32_t cimport LLSparseMatrixView_INT32_t_INT32_t



#from cysparse.sparse.csr_mat cimport MakeCSRSparseMatrix, MakeCSRComplexSparseMatrix
#from cysparse.sparse.csc_mat cimport MakeCSCSparseMatrix
#from cysparse.utils.equality cimport values_are_equal
#from cysparse.sparse.IO.mm cimport MakeLLSparseMatrixFromMMFile2, MakeMMFileFromSparseMatrix

from cysparse.sparse.sparse_utils.generate_indices_INT32_t cimport create_c_array_indices_from_python_object_INT32_t

########################################################################################################################
# CySparse include
########################################################################################################################
# pxi files should come last (except for circular dependencies)
include "ll_mat_kernel/ll_mat_assignment_kernel_INT32_t_INT32_t.pxi"
include "ll_mat_kernel/ll_mat_multiplication_by_numpy_vector_kernel_INT32_t_INT32_t.pxi"
include "ll_mat_helpers/ll_mat_multiplication_INT32_t_INT32_t.pxi"


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
    PyObject* PyFloat_FromDouble(double v)
    Py_complex PyComplex_AsCComplex(PyObject *op)

cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)
    double creal(double complex z)
    double cimag(double complex z)
    long double creall(long double complex z)
    long double cimagl(long double complex z)

########################################################################################################################
# CySparse cimport/import to avoid circular dependencies
########################################################################################################################
from cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_INT32_t cimport LLSparseMatrixView_INT32_t_INT32_t, MakeLLSparseMatrixView_INT32_t_INT32_t


########################################################################################################################
# CLASS LLSparseMatrix
########################################################################################################################
cdef class LLSparseMatrix_INT32_t_INT32_t(MutableSparseMatrix_INT32_t_INT32_t):
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
        self.type_name = "LLSparseMatrix [INT32_t, INT32_t]"

        # This is particular to the LLSparseMatrix type
        # Do we allocate memory here or
        # do we let another factory method do it for us?
        no_memory = kwargs.get('no_memory', False)

        cdef INT32_t i

        if not no_memory:

            val = <INT32_t *> PyMem_Malloc(self.size_hint * sizeof(INT32_t))
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

        temp = <INT32_t *> PyMem_Realloc(self.val, nalloc_new * sizeof(INT32_t))
        if not temp:
            raise MemoryError()
        self.val = <INT32_t *>temp

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
        total_memory += self.nalloc * sizeof(INT32_t)

        return total_memory

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
    cdef assign(self, LLSparseMatrixView_INT32_t_INT32_t view, object obj):
        """
        Set ``A[..., ...] = obj`` directly.

        Args:
            view: An ``LLSparseMatrixView_INT32_t_INT32_t`` that points to this matrix (``self``).
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
                        self.put(row_indices[i], col_indices[j], <INT32_t> obj[tuple(i, j)])

        else:   # self.is_symmetric == False

            if PyLLSparseMatrix_Check(obj):
                # obj is LLSparseMatrix
                for i from 0 <= i < nrow:
                    for j from 0 <= j < ncol:
                        self.put(row_indices[i], col_indices[j], obj.at(i, j))

            else:
                for i from 0 <= i < nrow:
                    for j from 0 <= j < ncol:
                        self.put(row_indices[i], col_indices[j], <INT32_t> obj[tuple(i, j)])

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
    cdef put(self, INT32_t i, INT32_t j, INT32_t value):
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


    cdef int safe_put(self, INT32_t i, INT32_t j, INT32_t value)  except -1:
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
    cdef INT32_t at(self, INT32_t i, INT32_t j):
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

    cdef INT32_t safe_at(self, INT32_t i, INT32_t j) except? 1:

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

        cdef LLSparseMatrixView_INT32_t_INT32_t view

        # test for direct access (i.e. both elements are integers)
        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            view =  MakeLLSparseMatrixView_INT32_t_INT32_t(self, <PyObject *>key[0], <PyObject *>key[1])
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

                    PyList_SET_ITEM(list_p, pos, Py_BuildValue("i", self.val[k]))


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
            INT32_t val

        list_p = PyList_New(self.nnz)
        if list_p == NULL:
            raise MemoryError()

        # EXPLICIT TYPE TESTS

        for i from 0 <= i < self.nrow:
            k = self.root[i]
            while k != -1:
                j = self.col[k]
                val = self.val[k]

                PyList_SET_ITEM(list_p, pos, Py_BuildValue("((ii)i)", i, j, self.val[k]))


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



            cnp.ndarray[cnp.int32_t, ndim=1] a_val = cnp.PyArray_SimpleNew( 1, dmat, cnp.NPY_INT32)


            INT32_t   *pi, *pj;   # Intermediate pointers to matrix data
            INT32_t    *pv;
            INT32_t   i, k, elem;

        pi = <INT32_t *> cnp.PyArray_DATA(a_row)
        pj = <INT32_t *> cnp.PyArray_DATA(a_col)
        pv = <INT32_t *> cnp.PyArray_DATA(a_val)

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
    # Multiplication
    ####################################################################################################################
    def matvec(self, B):
        """
        Return :math:`A * b`.
        """
        return multiply_ll_mat_with_numpy_vector_INT32_t_INT32_t(self, B)

    def matvec_transp(self, B):
        """
        Return :math:`A^t * b`.
        """
        return multiply_transposed_ll_mat_with_numpy_vector_INT32_t_INT32_t(self, B)

    def __mul__(self, B):
        """
        Classical matrix multiplication.

        Cases:

        - ``C = A * B`` where `B` is an ``LLSparseMatrix`` matrix. ``C`` is an ``LLSparseMatrix`` of same type.
        - ``C = A * B`` where ``B`` is an :program:`NumPy` matrix. ``C`` is a dense :program:`NumPy` matrix. (not yet implemented).
        """
        # CASES
        if PyLLSparseMatrix_Check(B):
            return multiply_two_ll_mat_INT32_t_INT32_t(self, B)
            #raise NotImplementedError("Multiplication with this kind of object not implemented yet...")
        elif cnp.PyArray_Check(B):
            # test type
            assert are_mixed_types_compatible(INT32_T, B.dtype), "Multiplication only allowed with a Numpy compatible type (%s)!" % cysparse_to_numpy_type(INT32_T)

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


    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def print_to(self, OUT, width=9, print_big_matrices=False):
        """
        Print content of matrix to output stream.

        Args:
            OUT: Output stream that print (Python3) can print to.

        """
        # TODO: adapt to any numbers... and allow for additional parameters to control the output
        # TODO: don't create temporary matrix
        cdef INT32_t i, k, first = 1

        print(self._matrix_description_before_printing(), file=OUT)

        cdef INT32_t *mat
        cdef INT32_t j
        cdef INT32_t val, ival

        if not self.nnz:
            return

        if print_big_matrices or (self.nrow <= LL_MAT_PPRINT_COL_THRESH and self.ncol <= LL_MAT_PPRINT_ROW_THRESH):
            # create linear vector presentation

            mat = <INT32_t *> PyMem_Malloc(self.nrow * self.ncol * sizeof(INT32_t))

            if not mat:
                raise MemoryError()

            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    # EXPLICIT TYPE TESTS

                    mat[i* self.ncol + j] = 0

                k = self.root[i]
                while k != -1:
                    mat[(i*self.ncol)+self.col[k]] = self.val[k]
                    k = self.link[k]

            # EXPLICIT TYPE TESTS
            for i from 0 <= i < self.nrow:
                for j from 0 <= j < self.ncol:
                    val = mat[(i*self.ncol)+j]

                    print('{:{width}.6f} '.format(val, width=width), end='', file=OUT)

                print(file=OUT)

            PyMem_Free(mat)

        else:
            print('Matrix too big to print out', file=OUT)