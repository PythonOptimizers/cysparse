from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

cdef enum:
    MM_MAX_LINE_LENGTH = 1025
    MM_MAX_TOKEN_LENGTH = 64

cdef enum:
    MM_PREMATURE_EOF = 12

# This is the way MM code its types
ctypedef char MM_typecode[4]


cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile(char * filename)