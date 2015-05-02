from cysparse.sparse.ll_mat cimport LLSparseMatrix


cpdef bint values_are_equal(double x, double y)

cpdef bint ll_mats_are_equals(LLSparseMatrix A, LLSparseMatrix B)