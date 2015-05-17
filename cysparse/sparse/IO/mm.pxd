from cysparse.sparse.ll_mat cimport LLSparseMatrix


cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile(str mm_filename)
cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile2(str mm_filename, bint store_zeros=?, bint test_bounds=?)

cdef MakeMMFileFromSparseMatrix(str mm_filename, LLSparseMatrix A)