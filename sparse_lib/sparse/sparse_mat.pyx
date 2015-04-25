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
        cdef INT_t memory = self.nrow * self.ncol * sizeof(double)
        return memory

    def memory_real(self):
        raise NotImplemented('Method not implemented for this type of matrix, please report')

    def memory_element(self):
        return sizeof(double)

cdef class MutableSparseMatrix(SparseMatrix):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!

        """
        self.size_hint = kwargs.get('size_hint', MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT)


cdef class ImmutableSparseMatrix(SparseMatrix):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        pass
