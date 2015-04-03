from sparse_lib.sparse.ll_mat import LLSparseMatrix

#class LLPySparseMatrix(LLCySparseMatrix):
#  pass


matrix = LLSparseMatrix(5, 5, 1)
print matrix
matrix[1,1] = 2
print matrix[1,1]
print matrix[0,4]
print matrix[2,2]

print matrix.nnz

try:
  matrix[100, 1000, 1] = -45343
except IndexError as e:
  print "value of error: %s" % e

try:
  matrix[100, 1000] = -45343
except IndexError as e:
  print "value of error: %s" % e


try:
  matrix[4, 4] = -45343
except IndexError as e:
  print "value of error: %s" % e


print matrix.nnz

matrix[1,1] = 9
matrix[2,2] = 10
matrix[3,3] = 11

print matrix.nnz

matrix[1,4] = 24323
matrix[2,4] = -876387263872

print matrix.nnz

print '=' * 80
print "element: " + str(matrix[4, 4])

print matrix[4, 4, 4]


