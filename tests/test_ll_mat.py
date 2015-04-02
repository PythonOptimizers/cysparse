from sparse_lib.sparse.ll_mat import LLSparseMatrix

#class LLPySparseMatrix(LLCySparseMatrix):
#  pass


matrix = LLSparseMatrix(5, 5, 3)
print matrix
matrix[1,1] = 2
print matrix[2,3]
