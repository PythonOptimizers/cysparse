"""
Write MatrixMarket format matrices.

See http://math.nist.gov/MatrixMarket/
"""
from sparse_lib.sparse.ll_mat cimport LLSparseMatrix


cdef MakeMMFileFromSparseMatrix(str mm_filename, LLSparseMatrix A):
    pass