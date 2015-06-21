"""
This file compares different implementations of ``generalize``, i.e. the transformation of a ``LLSparseMatrix`` from symmetrical to non symmetrical (in place).

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


    label = "Generalize norm with 100 elements and size = 1,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, __is_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)

    #def tearDown(self):

    #    assert self.A_c.nnz == self.A_p.nnz

    #    for i in xrange(self.size):
    #        for j in xrange(self.size):
    #            assert self.A_c[i,j] == self.A_p[i,j]

    def test_pysparse(self):
        self.A_p.generalize()
        return

    def test_cysparse(self):
        self.A_c.generalize()
        return


class LLMatFrobeniusNormBenchmark_1(LLMatFrobeniusNormBenchmark):


    label = "Generalize norm with 1,000 elements and size = 10,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, __is_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)


class LLMatFrobeniusNormBenchmark_2(LLMatFrobeniusNormBenchmark):


    label = "Generalize norm with 10,000 elements and size = 100,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, __is_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)

class LLMatFrobeniusNormBenchmark_3(LLMatFrobeniusNormBenchmark):


    label = "Generalize norm with 80,000 elements and size = 100,000 for a symmetrical matrix"
    each = 100


    def setUp(self):

        self.nbr_elements = 80000
        self.size = 100000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, __is_symmetric=True)
        construct_sym_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat_sym(self.size, self.size, self.nbr_elements)
        construct_sym_sparse_matrix(self.A_p, self.size, self.nbr_elements)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")