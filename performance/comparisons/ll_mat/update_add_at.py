"""
This file compares different implementations of update_add_at, i.e.

..  code-block:: python

    for i in range(len(val)):
        A[id1[i],id2[i]] += val[i]

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
class LLMatUpdateAddAtBenchmark(benchmark.Benchmark):


    label = "Simple update_add_at with 100 elements and size = 1,000 and 100 elements to add"
    each = 10


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000
        self.nbr_elements_to_add = 100

        assert self.nbr_elements_to_add < self.size

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.ones(self.nbr_elements_to_add, dtype=np.int32)
        self.id2 = self.v = np.arange(0, self.nbr_elements_to_add, dtype=np.int32)

        self.val = np.empty(self.nbr_elements_to_add, dtype=np.float64)

    #def tearDown(self):
    #    for i in xrange(self.size):
    #        for j in xrange(self.size):
    #            assert self.A_c[i,j] == self.A_p[i,j]

    def test_pysparse(self):
        self.A_p.update_add_at(self.val, self.id1, self.id2)
        return

    def test_cysparse(self):
        self.A_c.update_add_at(self.id1, self.id2, self.val)
        return


class LLMatUpdateAddAtBenchmark_1(LLMatUpdateAddAtBenchmark):


    label = "Simple update_add_at with 1,000 elements and size = 10,000 and 1,000 elements to add"
    each = 10


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000
        self.nbr_elements_to_add = 1000

        assert self.nbr_elements_to_add < self.size

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.ones(self.nbr_elements_to_add, dtype=np.int32)
        self.id2 = self.v = np.arange(0, self.nbr_elements_to_add, dtype=np.int32)

        self.val = np.empty(self.nbr_elements_to_add, dtype=np.float64)


class LLMatUpdateAddAtBenchmark_2(LLMatUpdateAddAtBenchmark):


    label = "Simple update_add_at with 10,000 elements and size = 100,000 and 10,000 elements to add"
    each = 10


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000
        self.nbr_elements_to_add = 10000

        assert self.nbr_elements_to_add < self.size

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.ones(self.nbr_elements_to_add, dtype=np.int32)
        self.id2 = self.v = np.arange(0, self.nbr_elements_to_add, dtype=np.int32)

        self.val = np.empty(self.nbr_elements_to_add, dtype=np.float64)


class LLMatUpdateAddAtBenchmark_3(LLMatUpdateAddAtBenchmark):


    label = "Simple update_add_at with 80,000 elements and size = 100,000 and 50,000 elements to add"
    each = 10


    def setUp(self):

        self.nbr_elements = 80000
        self.size = 100000
        self.nbr_elements_to_add = 50000

        assert self.nbr_elements_to_add < self.size

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.ones(self.nbr_elements_to_add, dtype=np.int32)
        self.id2 = self.v = np.arange(0, self.nbr_elements_to_add, dtype=np.int32)

        self.val = np.empty(self.nbr_elements_to_add, dtype=np.float64)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")