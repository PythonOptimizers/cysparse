"""
This file compares different implementations of the Frobenius norm for symmetrical matrices.

We compare the libraries:

- :program:`PySparse` and
- :program:`CySparse`.

"""

import benchmark
import numpy as np

# CySparse
from cysparse.sparse.ll_mat import LLSparseMatrix
from cysparse.common_types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

# PySparse
from pysparse.sparse import spmatrix


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


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatFrobeniusNormBenchmark(benchmark.Benchmark):


    label = "Frobenius norm with 100 elements and size = 1,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, store_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)

    def tearDown(self):

        assert self.p_norm == self.c_norm
            #assert self.w_c[i] == self.w_s[i]


    def test_pysparse(self):
        self.p_norm = self.A_p.norm('fro')
        return

    def test_cysparse(self):
        self.c_norm = self.A_c.norm('frob')
        return


class LLMatFrobeniusNormBenchmark_1(LLMatFrobeniusNormBenchmark):


    label = "Frobenius norm with 1,000 elements and size = 10,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, store_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)

class LLMatFrobeniusNormBenchmark_2(LLMatFrobeniusNormBenchmark):


    label = "Frobenius norm with 10,000 elements and size = 100,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, store_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)


class LLMatFrobeniusNormBenchmark_3(LLMatFrobeniusNormBenchmark):


    label = "Frobenius norm with 80,000 elements and size = 100,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 80000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, store_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")