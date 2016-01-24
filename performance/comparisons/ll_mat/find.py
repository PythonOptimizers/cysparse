"""
This file compares different implementations of ``find``.

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
def construct_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatFindBenchmark(benchmark.Benchmark):


    label = "Simple find with 100 elements, size = 1,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)


    #def tearDown(self):
    #    assert self.A_c.nnz == self.A_p.nnz

    #    # reconstruct initial matrices
    #    self.A_c_bis = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
    #    self.A_c_bis.put_triplet(self.A_c_rows, self.A_c_cols, self.A_c_vals)

    #    self.A_p_bis = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
    #    self.A_p_bis.put(self.A_p_vals, self.A_p_rows, self.A_p_cols)


    #    for i in xrange(self.size):
    #        for j in xrange(self.size):
    #            assert self.A_c_bis[i, j] == self.A_p_bis[i, j]

    def test_pysparse(self):
        self.A_p_vals, self.A_p_rows, self.A_p_cols = self.A_p.find()
        return

    def test_cysparse(self):
        self.A_c_rows, self.A_c_cols, self.A_c_vals = self.A_c.find()
        return


class LLMatFindBenchmark_1(LLMatFindBenchmark):


    label = "Simple find with 1,000 elements, size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)


class LLMatFindBenchmark_2(LLMatFindBenchmark):


    label = "Simple find with 10,000 elements, size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)


class LLMatFindBenchmark_3(LLMatFindBenchmark):


    label = "Simple find with 80,000 elements, size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 80000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")