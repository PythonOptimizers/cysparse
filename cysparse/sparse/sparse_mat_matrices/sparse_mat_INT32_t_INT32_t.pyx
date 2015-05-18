from cysparse.types.cysparse_types cimport *
from cysparse.sparse.sparse_mat cimport SparseMatrix, unexposed_value, MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

########################################################################################################################
# BASE MATRIX CLASS
########################################################################################################################
cdef class SparseMatrix_INT32_t_INT32_t(SparseMatrix):

    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        assert unexposed_value == kwargs.get('control_object', None), "Matrix must be instantiated with a factory method"

        self.type_name = "SparseMatrix_INT32_t"

        self.nrow = kwargs.get('nrow', -1)
        self.ncol = kwargs.get('ncol', -1)
        self.nnz = kwargs.get('nnz', 0)


    ####################################################################################################################
    # MEMORY INFO
    ####################################################################################################################
    def memory_virtual(self):
        """
        Return memory (in bits) needed if implementation would have kept **all** elements, not only the non zeros ones.

        Note:
            This method only returns the internal memory used for the C-arrays, **not** the whole object.
        """
        return INT32_t_BIT * self.nrow * self.ncol

    def memory_real(self):
        """
        Real memory used internally.

        Note:
            This method only returns the internal memory used for the C-arrays, **not** the whole object.
        """
        raise NotImplemented('Method not implemented for this type of matrix, please report')

    def memory_element(self):
        """
        Return memory used to store **one** element (in bits).


        """
        return INT32_t_BIT

    ####################################################################################################################
    # OUTPUT STRINGS
    ####################################################################################################################
    def attributes_short_string(self):
        """

        """
        s = "of dim %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
        return s

    def attributes_long_string(self):

        symmetric_string = None
        if self.is_symmetric:
            symmetric_string = 'symmetric'
        else:
            symmetric_string = 'general'

        store_zeros_string = None
        if self.store_zeros:
            store_zeros_string = "store_zeros"
        else:
            store_zeros_string = "no_zeros"

        s = "%s [%s, %s]" % (self.attributes_short_string(), symmetric_string, store_zeros_string)

        return s

    def attributes_condensed(self):
        symmetric_string = None
        if self.is_symmetric:
            symmetric_string = 'S'
        else:
            symmetric_string = 'G'

        store_zeros_string = None
        if self.store_zeros:
            store_zeros_string = "SZ"
        else:
            store_zeros_string = "NZ"

        s= "(%s, %s, [%d, %d])" % (symmetric_string, store_zeros_string, self.nrow, self.ncol)

        return s

    def _matrix_description_before_printing(self):
        s = "%s %s" % (self.type_name, self.attributes_condensed())
        return s

    def __repr__(self):
        s = "%s %s" % (self.type_name, self.attributes_long_string())
        return s

########################################################################################################################
# BASE MUTABLE MATRIX CLASS
########################################################################################################################
cdef class MutableSparseMatrix_INT32_t_INT32_t(SparseMatrix_INT32_t_INT32_t):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!

        """
        self.size_hint = kwargs.get('size_hint', MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT)
        self.nalloc = 0
        self.is_mutable = True


########################################################################################################################
# BASE IMMUTABLE MATRIX CLASS
########################################################################################################################
cdef class ImmutableSparseMatrix_INT32_t_INT32_t(SparseMatrix_INT32_t_INT32_t):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        self.is_mutable = False