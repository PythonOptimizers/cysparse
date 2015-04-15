"""
Tests of ``c = A * b`` with ``A`` a :class:`LLSparseMatrix` and ``b`` and ``c`` one-dimensional numpy arrays.

``A`` can be symmetric or not and ``b`` and ``c`` can be strided or not (C-contiguous or not).
"""

from sparse_lib.sparse.ll_mat import LLSparseMatrix
from sparse_lib.utils.equality import ll_mats_are_equals
import unittest

import numpy as np


class LLSparseMatrixMultiplicationWithNumpyVectorBaseTestCase(unittest.TestCase):
    def setUp(self):

        self.m = 3
        self.n = 2
        self.size_hint = 15
        self.A = LLSparseMatrix(nrow=self.m, ncol=self.n, size_hint=self.size_hint)
        self.A[0, 0] = 1
        self.A[0, 1] = 3.2
        self.A[1, 0] = 3.2
        self.A[1, 1] = 1

        self.A_sym = LLSparseMatrix(nrow=self.m, ncol=self.n, size_hint=self.size_hint, is_symmetric=True)
        self.A_sym[0, 0] = 1
        self.A_sym[1, 0] = 3.2
        self.A_sym[1, 1] = 1

        self.b = np.array([0 , 1]).astype(np.float64)  # C-contiguous

        self.b_strided = np.array([0.0, -32, 1, 89]).astype(np.float64)[::2] # non C-contiguous



class LLSparseMatrixMultiplicationWithNumpyVectorBasicTestCase(LLSparseMatrixMultiplicationWithNumpyVectorBaseTestCase):
    def test_both_ll_mat_are_equal(self):
        self.failUnless(ll_mats_are_equals(self.A, self.A_sym))

    def test_all_cases(self):
        c1 = self.A * self.b
        print c1

        c2 = self.A * self.b_strided
        print c2

        c3 = self.A_sym * self.b
        print c3

        c4 = self.A_sym * self.b_strided
        print c4


if __name__ == '__main__':
    unittest.main()
