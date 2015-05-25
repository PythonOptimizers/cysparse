from cysparse.types.cysparse_types cimport *
from cysparse.sparse.t_mat cimport TransposedSparseMatrix

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default

unexposed_value = object()


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
        assert unexposed_value == kwargs.get('control_object', None), "Matrix must be instantiated with a factory method"

        self.type_name = "Not defined"
        self.cp_type.itype = kwargs.get('itype', INT32_T)
        self.cp_type.dtype = kwargs.get('dtype', FLOAT64_T)

        self.is_symmetric = kwargs.get('is_symmetric', False)
        self.store_zeros = kwargs.get('store_zeros', False)
        self.is_mutable = False

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
            #raise NotImplementedError("Not implemented in base class")
            # create proxy
            return TransposedSparseMatrix(self)

        def __set__(self, value):
            raise AttributeError('Attribute T (transposed) is read-only')

        def __del__(self):
            raise AttributeError('Attribute T (transposed) is read-only')

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
