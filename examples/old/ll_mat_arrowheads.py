from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

A = NewArrowheadLLSparseMatrix(nrow=3, ncol=4, element=3.8)

A.print_to(sys.stdout)