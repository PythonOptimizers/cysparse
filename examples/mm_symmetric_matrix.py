#!/usr/bin/env python

from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

import sys


# youngX.mtx are NOT symmetric...
l2 = NewLLSparseMatrixFromMMFile('zenios.mtx')
l2.print_to(sys.stdout)