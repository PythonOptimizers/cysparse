from cysparse.types.cysparse_types cimport *
from cysparse.sparse.s_mat cimport SparseMatrix, unexposed_value, MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

from cysparse.sparse.sparse_proxies.t_mat cimport TransposedSparseMatrix

from cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT32_t_COMPLEX256_t cimport ConjugateTransposedSparseMatrix_INT32_t_COMPLEX256_t
from cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT32_t_COMPLEX256_t cimport ConjugatedSparseMatrix_INT32_t_COMPLEX256_t


from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

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

    PyObject* Py_BuildValue(char *format, ...)
    PyObject* PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(PyObject *list, Py_ssize_t i, PyObject *o)
    PyObject* PyList_GET_ITEM(PyObject *list, Py_ssize_t i)

########################################################################################################################
# BASE MATRIX CLASS
########################################################################################################################
cdef class SparseMatrix_INT32_t_COMPLEX256_t(SparseMatrix):

    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        assert unexposed_value == kwargs.get('control_object', None), "Matrix must be instantiated with a factory method"

        self.__index_and_type = "[INT32_t, COMPLEX256_t]"

        self.__type = "SparseMatrix"
        self.__type_name = "SparseMatrix %s" % self.__index_and_type

        self.cp_type.itype = INT32_T
        self.cp_type.dtype = COMPLEX256_T

        self.__nrow = kwargs.get('nrow', -1)
        self.__ncol = kwargs.get('ncol', -1)
        self.__nnz = kwargs.get('nnz', 0)

        self.__nargin = self.__ncol
        self.__nargout = self.__nrow

        self.__transposed_proxy_matrix_generated = False


        self.__conjugate_transposed_proxy_matrix_generated = False
        self.__conjugated_proxy_matrix_generated = False


    
    @property
    def nrow(self):
        return self.__nrow

    
    @property
    def ncol(self):
        return self.__ncol

    
    @property
    def nnz(self):
        return self.__nnz

    
    @property
    def nargin(self):
        return self.__nargin

    
    @property
    def nargout(self):
        return self.__nargout

    
    @property
    def T(self):
        if not self.__transposed_proxy_matrix_generated:
            # create proxy
            self.__transposed_proxy_matrix = TransposedSparseMatrix(self)
            self.__transposed_proxy_matrix_generated = True

        return self.__transposed_proxy_matrix



    
    @property
    def H(self):
        if not self.__conjugate_transposed_proxy_matrix_generated:
            # create proxy
            self.__conjugate_transposed_proxy_matrix = ConjugateTransposedSparseMatrix_INT32_t_COMPLEX256_t(self)
            self.__conjugate_transposed_proxy_matrix_generated = True

        return self.__conjugate_transposed_proxy_matrix
    
    @property
    def conj(self):
        if not self.__conjugated_proxy_matrix_generated:
            # create proxy
            self.__conjugated_proxy_matrix = ConjugatedSparseMatrix_INT32_t_COMPLEX256_t(self)
            self.__conjugated_proxy_matrix_generated = True

        return self.__conjugated_proxy_matrix


    ####################################################################################################################
    # Set/Get list of elements
    ####################################################################################################################
    ####################################################################################################################
    #                                            *** SET ***
    ####################################################################################################################
    #                                            *** GET ***
    def diags(self, diag_coeff):
        """
        Return a list wiht :program:`NumPy` arrays containings asked diagonals.

        Args:
            diag_coeff: Can be a list or a slice.

        Raises:
            - ``RuntimeError`` if slice is illformed;
            - ``TypeError`` if argument is not a ``list`` or ``slice``;
            - ``MemoryError`` if there is not enough memory for internal calculations;
            - ``ValueError`` if the list contains something else than integer indices;
            - ``AssertionError`` if internal calculations go wrong (should not happen...);
            - ``IndexError`` if the diagonals coefficients are out of bound.

        Note:
            Diagonal coefficients greater than ``n-1`` as disregarded when using a slice.
        """
        cdef INT32_t ret
        cdef Py_ssize_t start, stop, step, length, index, max_length

        cdef INT32_t i, j
        cdef INT32_t * indices
        cdef PyObject *val

        cdef PyObject * obj = <PyObject *> diag_coeff

        # normally, with slices, it is common in Python to chop off...
        # Here we only chop off from above, not below...
        # -m + 1 <= k <= n -1   : only k <= n - 1 will be satified (greater indices are disregarded)
        # but nothing is done if k < -m + 1
        max_length = self.__ncol

        # grab diag coefficients
        if PySlice_Check(obj):
            # slice
            ret = PySlice_GetIndicesEx(<PySliceObject*>obj, max_length, &start, &stop, &step, &length)
            if ret:
                raise RuntimeError("Slice could not be translated")

            #print "start, stop, step, length = (%d, %d, %d, %d)" % (start, stop, step, length)

            indices = <INT32_t *> PyMem_Malloc(length * sizeof(INT32_t))
            if not indices:
                raise MemoryError()

            # populate indices
            i = start
            for j from 0 <= j < length:
                indices[j] = i
                i += step

        elif PyList_Check(obj):
            length = PyList_Size(obj)
            indices = <INT32_t *> PyMem_Malloc(length * sizeof(INT32_t))
            if not indices:
                raise MemoryError()

            for i from 0 <= i < length:
                val = PyList_GetItem(obj, <Py_ssize_t>i)
                if PyInt_Check(val):
                    index = PyInt_AS_LONG(val)
                    indices[i] = <INT32_t> index
                else:
                    PyMem_Free(indices)
                    raise ValueError("List must only contain integers")
        else:
            raise TypeError("Index object is not recognized (list or slice)")

        diagonals = list()

        for i from 0 <= i < length:
            diagonals.append(self.diag(indices[i]))

        return diagonals

    ####################################################################################################################
    # CREATE SPECIAL MATRICES
    ####################################################################################################################
    def create_transpose(self):
        raise NotImplementedError('Not yet implemented. Please report.')


    def create_conjugate_transpose(self):
        raise NotImplementedError('Not yet implemented. Please report.')

    def create_conjugate(self):
        raise NotImplementedError('Not yet implemented. Please report.')


    ####################################################################################################################
    # MEMORY INFO
    ####################################################################################################################
    def memory_virtual(self):
        """
        Return memory (in bits) needed if implementation would have kept **all** elements, not only the non zeros ones.

        Note:
            This method only returns the internal memory used for the C-arrays, **not** the whole object.
        """
        return COMPLEX256_t_BIT * self.__nrow * self.__ncol

    def memory_real(self):
        """
        Real memory used internally.

        Note:
            This method only returns the internal memory used for the C-arrays, **not** the whole object.
        """
        raise NotImplementedError('Method not implemented for this type of matrix, please report')

    def memory_element(self):
        """
        Return memory used to store **one** element (in bits).


        """
        return COMPLEX256_t_BIT

    ####################################################################################################################
    # OUTPUT STRINGS
    ####################################################################################################################
    def attributes_short_string(self):
        """

        """
        s = "of dim (%d, %d) with %d non zero values" % (self.__nrow, self.__ncol, self.__nnz)
        return s

    def attributes_long_string(self):

        symmetric_string = None
        if self.__is_symmetric:
            symmetric_string = 'symmetric'
        else:
            symmetric_string = 'general'

        store_zeros_string = None
        if self.__store_zeros:
            store_zeros_string = "store_zeros"
        else:
            store_zeros_string = "no_zeros"

        s = "%s [%s, %s]" % (self.attributes_short_string(), symmetric_string, store_zeros_string)

        return s

    def attributes_condensed(self):
        symmetric_string = None
        if self.__is_symmetric:
            symmetric_string = 'S'
        else:
            symmetric_string = 'G'

        store_zeros_string = None
        if self.__store_zeros:
            store_zeros_string = "SZ"
        else:
            store_zeros_string = "NZ"

        s= "(%s, %s, [%d, %d])" % (symmetric_string, store_zeros_string, self.__nrow, self.__ncol)

        return s

    def _matrix_description_before_printing(self):
        s = "%s %s" % (self.__type_name, self.attributes_condensed())
        return s

    def __repr__(self):
        s = "%s %s" % (self.__type_name, self.attributes_long_string())
        return s

########################################################################################################################
# BASE MUTABLE MATRIX CLASS
########################################################################################################################
cdef class MutableSparseMatrix_INT32_t_COMPLEX256_t(SparseMatrix_INT32_t_COMPLEX256_t):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!

        """
        self.size_hint = kwargs.get('size_hint', MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT)

        # test to bound memory allocation
        if self.size_hint > self.nrow * self.ncol:
            self.size_hint = self.nrow *  self.ncol

        self.nalloc = 0
        self.__is_mutable = True


########################################################################################################################
# BASE IMMUTABLE MATRIX CLASS
########################################################################################################################
cdef class ImmutableSparseMatrix_INT32_t_COMPLEX256_t(SparseMatrix_INT32_t_COMPLEX256_t):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        self.__is_mutable = False