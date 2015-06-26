from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

A = NewLLSparseMatrixArrowhead(nrow=3, ncol=4, element=3.8)

A.print_to(sys.stdout)