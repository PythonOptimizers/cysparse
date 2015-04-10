

cdef class SparseMatrix:

    def __cinit__(self, *args, **kwargs):
        pass


cdef class MutableSparseMatrix(SparseMatrix):
    def __cinit__(self, *args, **kwargs):
        pass


cdef class ImmutableSparseMatrix(SparseMatrix):
    def __cinit__(self, *args, **kwargs):
        pass
