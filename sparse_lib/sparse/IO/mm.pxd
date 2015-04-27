from sparse_lib.sparse.ll_mat cimport LLSparseMatrix


cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile(str filename)
cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile2(str filename, bint store_zeros=?, bint test_bounds=?)