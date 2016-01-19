#!/usr/bin/env python

"""
This file tests the global ``matvec`` functions.

We tests:

- ``matvec()``;
- ``matvec_transp()``;
- ``matvec_adj()``;
- ``matvec_conj()``;


"""

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

############################################
# matvec
############################################
class CySparseGlobalMatVecFunction_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT32_T, itype=INT32_T)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.float32)

        self.x.fill(2)


        # sparse vector
        self.v_ll_mat = LinearFillLLSparseMatrix(nrow=self.ncol, ncol=1, dtype=FLOAT32_T, itype=INT32_T)


        self.C = self.A.to_csc()
        self.v = self.v_ll_mat.to_csc()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A * self.x
        result_with_C = matvec(self.C, self.x)

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])




############################################
# matvec_transp
############################################
class CySparseGlobalMatVecTranspFunction_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT32_T, itype=INT32_T)

        # numpy vectors
        self.x = np.empty(self.nrow, dtype=np.float32)

        self.x.fill(2)


        # sparse vector
        self.v_ll_mat = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=1, dtype=FLOAT32_T, itype=INT32_T)


        self.C = self.A.to_csc()
        self.v = self.v_ll_mat.to_csc()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec_transp`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A.T * self.x
        result_with_C = matvec_transp(self.C, self.x)

        for i in range(self.ncol):
            self.assertTrue(result_with_A[i] == result_with_C[i])



############################################
# matvec_adj
############################################
class CySparseGlobalMatVecHTranspFunction_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT32_T, itype=INT32_T)

        # numpy vectors
        self.x = np.empty(self.nrow, dtype=np.float32)

        self.x.fill(2)



        self.C = self.A.to_csc()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec_adj`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A.H * self.x
        result_with_C = matvec_adj(self.C, self.x)

        for i in range(self.ncol):
            self.assertTrue(result_with_A[i] == result_with_C[i])


############################################
# matvec_conj
############################################
class CySparseGlobalMatVecConjFunction_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT32_T, itype=INT32_T)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.float32)

        self.x.fill(2)



        self.C = self.A.to_csc()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec_conj`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A.conj * self.x
        result_with_C = matvec_conj(self.C, self.x)

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])


if __name__ == '__main__':
    unittest.main()