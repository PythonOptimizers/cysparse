from sparse_lib.sparse.ll_mat cimport LLSparseMatrix


cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile(str mm_filename)
cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile2(str mm_filename, bint store_zeros=?, bint test_bounds=?)