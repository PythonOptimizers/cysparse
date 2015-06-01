"""
Test ``LLSparseMatrix`` norms.
"""

from cysparse.types.cysparse_types import *
from cysparse.sparse.ll_mat import *

# PySparse
from pysparse.sparse import spmatrix

import unittest


########################################################################################################################
# Helpers
########################################################################################################################
def construct_sym_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        k = i % n
        p = (i % 2 + 1) % n
        if k >= p:
            A[k, p] = i / 3
        else:
            A[p, k] = i / 3


class CySparseLLSparseMatrixNormsBaseTestCase(unittest.TestCase):
    def setUp(self):
        pass


class CySparseSymmetricLLSparseMatrixNormsTestCase(CySparseLLSparseMatrixNormsBaseTestCase):
    """
    Test norms for symmetric matrices.
    """
    def test_norm_values(self):
        self.nbr_elements = 100
        self.size = 1000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, is_symmetric=True)
        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)

        for i in xrange(self.nbr_elements):
            k = i % self.size
            p = (i % 2 + 1) % self.size
            if k >= p:
                self.A_c[k, p] = i / 3
                self.A_p[k, p] = i / 3
            else:
                self.A_c[p, k] = i / 3
                self.A_p[p, k] = i / 3

            self.failUnless(self.A_p.norm('fro') == self.A_c.norm('frob'))
            self.failUnless(self.A_c.norm('1') == self.A_c.norm('inf'))

        self.A_p.generalize()
        self.A_c.generalize()

        self.failUnless(self.A_p.norm('fro') == self.A_c.norm('frob'))
        self.failUnless(self.A_p.norm('1') == self.A_c.norm('1'))
        self.failUnless(self.A_p.norm('inf') == self.A_c.norm('inf'))


class CySparseLLSparseMatrixNormsTestCase(CySparseLLSparseMatrixNormsBaseTestCase):
    """
    Test norms for non symmetric matrices.
    """
    def test_norm_values(self):
        self.nbr_elements = 100
        self.size = 1000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)

        for i in xrange(self.nbr_elements):
            self.A_c[i % self.size, (2 * i + 1) % self.size] = i / 3
            self.A_p[i % self.size, (2 * i + 1) % self.size] = i / 3

            self.failUnless(self.A_p.norm('fro') == self.A_c.norm('frob'))
            self.failUnless(self.A_p.norm('1') == self.A_c.norm('1'))
            self.failUnless(self.A_p.norm('inf') == self.A_c.norm('inf'))



if __name__ == '__main__':
    unittest.main()


