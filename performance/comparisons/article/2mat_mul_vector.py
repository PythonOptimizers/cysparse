"""
This file compares the multiplication of 2 matrices (CSR * CSC) multiplied by a NumPy vector.

We compare the libraries:

- :program:`CySparse` and
- :program:`SciPy.sparse`


"""

import benchmark
import numpy as np

import random as rd

# CySparse
from cysparse.sparse.ll_mat import LLSparseMatrix
from cysparse.common_types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

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


    label = "CSR * CSC * v with 1000 elements and size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSR_c = self.A_c.to_csr()
        self.CSR_s = self.A_s.tocsr()

        self.B_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.B_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.B_c)
        self.list_of_matrices.append(self.B_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSC_c = self.B_c.to_csc()
        self.CSC_s = self.B_s.tocsc()


        self.v = np.arange(0, self.size, dtype=np.float64)

    #def tearDown(self):
    #    for i in xrange(self.size):
    #        assert self.w_c[i] == self.w_p[i]
    #        assert self.w_c[i] == self.w_s[i]


    def test_cysparse(self):
        self.w_c = self.CSR_c * self.CSC_c * self.v
        return

    def test_scipy_sparse(self):
        self.w_s = self.CSR_s * self.CSC_s * self.v
        return



class LLMatMatVecBenchmark_2(LLMatMatVecBenchmark):


    label = "matvec with 10,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSR_c = self.A_c.to_csr()
        self.CSR_s = self.A_s.tocsr()

        self.B_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.B_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.B_c)
        self.list_of_matrices.append(self.B_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSC_c = self.B_c.to_csc()
        self.CSC_s = self.B_s.tocsc()

        self.v = np.arange(0, self.size, dtype=np.float64)


class LLMatMatVecBenchmark_3(LLMatMatVecBenchmark):


    label = "matvec with 100,000 elements and size = 1,000,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100000
        self.size = 1000000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSR_c = self.A_c.to_csr()
        self.CSR_s = self.A_s.tocsr()

        self.B_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.B_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.B_c)
        self.list_of_matrices.append(self.B_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSC_c = self.B_c.to_csc()
        self.CSC_s = self.B_s.tocsc()

        self.v = np.arange(0, self.size, dtype=np.float64)


class LLMatMatVecBenchmark_4(LLMatMatVecBenchmark):


    label = "matvec with 5000 elements and size = 1,000,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 5000
        self.size = 1000000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.A_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.A_c)
        self.list_of_matrices.append(self.A_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSR_c = self.A_c.to_csr()
        self.CSR_s = self.A_s.tocsr()

        self.B_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        self.B_s = lil_matrix((self.size, self.size), dtype=np.float64)

        self.list_of_matrices = []
        self.list_of_matrices.append(self.B_c)
        self.list_of_matrices.append(self.B_s)

        construct_random_matrices(self.list_of_matrices, self.size, self.nbr_elements)

        self.CSC_c = self.B_c.to_csc()
        self.CSC_s = self.B_s.tocsc()
        self.v = np.arange(0, self.size, dtype=np.float64)

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")