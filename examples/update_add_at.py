from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=10)

id1 = np.array([1,1,1], dtype=np.int32)
id2 = np.array([0,1,2], dtype=np.int32)
val = np.array([1.6,1.7,1.8], dtype=np.float64)

########################################################################################################################
print "=" * 80

l1.update_add_at(id1, id2, val)

l1.print_to(sys.stdout)