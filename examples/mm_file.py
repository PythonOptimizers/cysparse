#!/usr/bin/env python

from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

import sys

if __name__ == "__main__":

    l1 = NewLLSparseMatrix(mm_filename='togolo.mtx', dtype=types.COMPLEX128_T, test_bounds=True)


    l1.print_to(sys.stdout)

