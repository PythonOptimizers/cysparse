from cysparse.types.cysparse_types cimport *
from cysparse.sparse.s_mat cimport SparseMatrix, unexposed_value, MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

from cysparse.sparse.sparse_proxies.t_mat cimport TransposedSparseMatrix

from cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT32_t_COMPLEX256_t cimport ConjugateTransposedSparseMatrix_INT32_t_COMPLEX256_t
from cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT32_t_COMPLEX256_t cimport ConjugatedSparseMatrix_INT32_t_COMPLEX256_t


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