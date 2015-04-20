"""
Test the :class:`NonZeros` context manager.
"""

from sparse_lib.sparse.ll_mat import MakeLLSparseMatrix
from sparse_lib.sparse.sparse_mat import NonZeros

import unittest


class NonZerosContextManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.m = 4
        self.n = 3
        self.size_hint = 15
        self.A = MakeLLSparseMatrix(nrow=3, ncol=4, store_zeros=True, size_hint=self.size_hint)
        self.A[0, 0] = 4.98
        self.A[2, 1] = -34.09

        self.nnz = 2

    def test_basic_use(self):

        self.assertTrue(self.A.store_zeros)

        with NonZeros(self.A):
            self.assertFalse(self.A.store_zeros)

        self.assertTrue(self.A.store_zeros)

    def test_number_of_elements(self):

        self.assertTrue(self.A.nnz == self.nnz)

        with NonZeros(self.A):
            self.A[0, 0] = 0
            self.A[2, 1] = 0.0
            self.A[1, 1] = -0

        self.assertTrue(self.A.nnz == 0)

if __name__ == '__main__':
    unittest.main()


