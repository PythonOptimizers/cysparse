from sparse_lib.cysparse_types cimport *

cdef INT_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default

class NonZeros():
    """
    Context manager to use methods with flag ``store_zeros`` to ``False``.

    The initial value is restored upon completion.

    Use like this:

        >>>with NonZeros(A):
        >>>    ...
    """
    def __init__(self, SparseMatrix A):
        self.A = A
        self.store_zeros = False

    def __enter__(self):
        self.store_zeros = self.A.store_zeros
        self.A.store_zeros = False

    def __exit__(self, type, value, traceback):
        self.A.store_zeros = self.store_zeros

########################################################################################################################
# BASE MATRIX CLASS
########################################################################################################################
cdef class SparseMatrix:

    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        self.nrow = kwargs.get('nrow', -1)
        self.ncol = kwargs.get('ncol', -1)
        self.nnz = kwargs.get('nnz', 0)

        self.is_symmetric = kwargs.get('is_symmetric', False)
        self.store_zeros = kwargs.get('store_zeros', False)
        self.is_complex = kwargs.get('is_complex', False)

    # for compatibility with numpy, array, etc
    property shape:
        def __get__(self):
            self.shape = (self.nrow, self.ncol)
            return self.shape

        def __set__(self, value):
            raise AttributeError('Attribute shape is read-only')

        def __del__(self):
            raise AttributeError('Attribute shape is read-only')

    property T:
        def __get__(self):
            raise NotImplemented("Not implemented in base class")

        def __set__(self, value):
            raise NotImplemented("Not implemented in base class")

        def __del__(self):
            raise NotImplemented("Not implemented in base class")

    ####################################################################################################################
    # MEMORY INFO
    ####################################################################################################################
    def memory_virtual(self):
        """
        Return memory needed if implementation would keep **all** elements, not only the non zeros ones.

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if self.is_complex:
            return COMPLEX_t_BIT * self.nrow * self.ncol
        else:
            return FLOAT_t_BIT * self.nrow * self.ncol

    def memory_real(self):
        raise NotImplemented('Method not implemented for this type of matrix, please report')

    def memory_element(self):
        """
        Return memory used to store **one** element (in bits).

        {{COMPLEX: YES}}
        {{GENERIC TYPES: YES}}
        """
        if self.is_complex:
            return COMPLEX_t_BIT
        else:
            return FLOAT_t_BIT

    def attributes_short_string(self):
        s = "of size %d by %d with %d non zero values" % (self.nrow, self.ncol, self.nnz)
        return s

    def attributes_long_string(self):

        symmetric_string = None
        if self.is_symmetric:
            symmetric_string = 'symmetric'
        else:
            symmetric_string = 'general'

        type_string = None
        if self.is_complex:
            type_string = "complex"
        else:
            type_string = "real"

        store_zeros_string = None
        if self.store_zeros:
            store_zeros_string = "store_zeros"
        else:
            store_zeros_string = "no_zeros"

        s = "%s [%s, %s, %s]" % (self.attributes_short_string(), symmetric_string, type_string, store_zeros_string)

        return s

    def attributes_condensed(self):
        symmetric_string = None
        if self.is_symmetric:
            symmetric_string = 'S'
        else:
            symmetric_string = 'G'

        type_string = None
        if self.is_complex:
            type_string = "C"
        else:
            type_string = "R"

        store_zeros_string = None
        if self.store_zeros:
            store_zeros_string = "SZ"
        else:
            store_zeros_string = "NZ"

        s= "(%s, %s, %s, [%d, %d])" % (symmetric_string, type_string, store_zeros_string, self.nrow, self.ncol)

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
cdef class MutableSparseMatrix(SparseMatrix):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!

        """
        self.size_hint = kwargs.get('size_hint', MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT)
        self.nalloc = 0


########################################################################################################################
# BASE IMMUTABLE MATRIX CLASS
########################################################################################################################
cdef class ImmutableSparseMatrix(SparseMatrix):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        pass
