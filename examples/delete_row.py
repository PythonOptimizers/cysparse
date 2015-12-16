from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLinearFillLLSparseMatrix(ncol=10, nrow=10)
print l1
########################################################################################################################
print "=" * 80

#mask = np.zeros((10,), dtype=np.int8)
#print mask.size
#mask[4] = 1
#mask[9] = 0

#l1.delete_rows_with_mask(mask)

#l1.print_to(sys.stdout)
#l1[0, 0] = types.nan
#print l1

########################################################################################################################
print "=" * 80

l1.delete_rows([2, 3, 0])

print l1