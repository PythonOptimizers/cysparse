"""
This file compares different implementations of ``matvec``, i.e. :math:`A * x`.

We compare the libraries:

- :program:`PySparse`;
- :program:`CySparse` and
- :program:`SciPy.sparse`

This time we specifically order the matrices before the multiplication.

"""

import benchmark
import numpy as np

import random as rd

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
def construct_random_matrices(list_of_matrices, n, nbr_elements):

    nbr_added_elements = 0

    A = list_of_matrices[0]

    while nbr_added_elements != nbr_elements:

        random_index1 = rd.randint(0, n - 1)
        random_index2 = rd.randint(0, n - 1)
        random_element = rd.uniform(0, 100)

        # test if element exists
        if A[random_index1, random_index2] != 0.0:
            continue

        for matrix in list_of_matrices:
            matrix[random_index1, random_index2] = random_element

        nbr_added_elements += 1


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatMatVecBenchmark(benchmark.Benchmark):


    label = "matvec with 1000 elements and size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_p)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)

    #def tearDown(self):
    #    for i in xrange(self.size):
    #        assert self.w_c[i] == self.w_p[i]
    #        assert self.w_c[i] == self.w_s[i]


    def test_pysparse(self):
        self.w_p = np.empty(self.size, dtype=np.float64)
        self.A_p.matvec(self.v, self.w_p)
        return

    def test_cysparse(self):
        self.w_c = self.A_c * self.v
        return

    def test_cysparse2(self):
        self.w_c2 = self.A_c.matvec(self.v)
        return

    def test_scipy_sparse(self):
        self.w_s = self.A_s * self.v
        return

    def test_scipy_sparse2(self):
        self.w_s2 = self.A_s._mul_vector(self.v)


class LLMatMatVecBenchmark_2(LLMatMatVecBenchmark):


    label = "matvec with 10,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_p)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)


class LLMatMatVecBenchmark_3(LLMatMatVecBenchmark):


    label = "matvec with 100,000 elements and size = 1,000,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100000
        self.size = 1000000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_p)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)


class LLMatMatVecBenchmark_4(LLMatMatVecBenchmark):


    label = "matvec with 5000 elements and size = 1,000,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 5000
        self.size = 1000000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_p)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")
