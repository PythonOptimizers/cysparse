
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

    # for compatibility with numpy, array, etc
    property shape:
        def __get__(self):
            self.shape = (self.nrow, self.ncol)
            return self.shape

        def __set__(self, value):
            raise AttributeError('Attribute shape is read-only')

        def __del__(self):
            raise AttributeError('Attribute shape is read-only')

cdef class MutableSparseMatrix(SparseMatrix):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!

        """
        self.size_hint = kwargs.get('size_hint', MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT)
        print("size hint = %d" % self.size_hint )

cdef class ImmutableSparseMatrix(SparseMatrix):
    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        pass
