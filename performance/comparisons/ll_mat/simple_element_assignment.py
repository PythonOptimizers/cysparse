"""
This file compares different implementations of simple element assignment, i.e. :math:`A[i, j] = x`.

We compare the libraries:

- :program:`PySparse`;
- :program:`CySparse` and
- :program:`SciPy.sparse`


"""

import benchmark
import numpy as np

# CySparse
from cysparse.sparse.ll_mat import LLSparseMatrix
from cysparse.common_types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

# PySparse
from pysparse.sparse import spmatrix

# SciPy
from scipy.sparse import lil_matrix

########################################################################################################################
# Helpers
########################################################################################################################
def construct_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatSimpleElementAssignemntBenchmark(benchmark.Benchmark):


    label = "Simple element assignment with 1000 elements and size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64) # how do we reserve space in advance?
        construct_sparse_matrix(self.A_s, self.size, self.nbr_elements)


    def test_pysparse(self):
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)
        return

    def test_cysparse(self):
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)
        return

    def test_scipy_sparse(self):
        construct_sparse_matrix(self.A_s, self.size, self.nbr_elements)
        return

class LLMatSimpleElementAssignemntBenchmark_1(LLMatSimpleElementAssignemntBenchmark):


    label = "Simple element assignment with 10,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64) # how do we reserve space in advance?

class LLMatSimpleElementAssignemntBenchmark_3(LLMatSimpleElementAssignemntBenchmark):


    label = "Simple element assignment with 100,000 elements and size = 1,000,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100000
        self.size = 1000000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64) # how do we reserve space in advance?

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")