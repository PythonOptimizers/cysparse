"""
This file compares different implementations of the infinite norm, i.e. :math:`||A||_\infty`.

We compare the libraries:

- :program:`PySparse` and
- :program:`CySparse`.


"""

import benchmark
import numpy as np

# CySparse
from cysparse.sparse.ll_mat import NewLLSparseMatrix
from cysparse.types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

# PySparse
from pysparse.sparse import spmatrix



########################################################################################################################
# Helpers
########################################################################################################################
def construct_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatInfiniteNormBenchmark(benchmark.Benchmark):


    label = "Infinite norm with 100 elements and size = 1,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

    def tearDown(self):
        assert self.p_norm_inf == self.c_norm_inf
            #assert self.w_c[i] == self.w_s[i]


    def test_pysparse(self):
        self.p_norm_inf = self.A_p.norm('inf')
        return

    def test_cysparse(self):
        self.c_norm_inf = self.A_c.norm('inf')
        return

class LLMatInfiniteNormBenchmark_1(LLMatInfiniteNormBenchmark):


    label = "Infinite norm with 1000 elements and size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

class LLMatInfiniteNormBenchmark_2(LLMatInfiniteNormBenchmark):


    label = "Infinite norm with 10,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

class LLMatInfiniteNormBenchmark_3(LLMatInfiniteNormBenchmark):


    label = "Infinite norm with 80,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 80000
        self.size = 100000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")