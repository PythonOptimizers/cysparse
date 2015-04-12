
cdef int MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default

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
