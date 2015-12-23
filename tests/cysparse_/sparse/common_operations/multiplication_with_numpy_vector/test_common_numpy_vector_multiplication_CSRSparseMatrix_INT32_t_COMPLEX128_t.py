#!/usr/bin/env python

"""
This file tests the basic multiplication operations with a :program:`NumPy` vector for **all** matrix like objects.

Proxies are only tested for a :class:`LLSparseMatrix` object.

We tests:

- Matrix-like objects:
    * Sparse Matrices;
    * Sparse Matrix Proxies;

- Optimized versions:
    * with/without Symmetry;
    * with/without StoreZero scheme;

- Numpy vectors:
    * with/without strides;

- Vector multiplication operations:
    * ``matvec()``;
    * ``matvec_transp()``;

and this for all combinations of indices and element types.

See file ``sparse_matrix_like_vector_multiplication``.

Note:
    We test ``matvec()`` through the ``*`` operator (syntactic sugar).
"""
from sparse_matrix_like_vector_multiplication import common_matrix_like_vector_multiplication

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

########################################################################################################################
# WITHOUT STRIDES IN THE NUMPY VECTORS
########################################################################################################################
##################################
# Case Non Symmetric, Non Zero
##################################
class CySparseCommonNumpyVectorMultiplication_CSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX128_T, itype=INT32_T)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.complex128)
        self.y = np.empty(self.nrow, dtype=np.complex128)

        self.x.fill(2 + 2j)
        self.y.fill(2 + 2j)



        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test ``matvec`` through the ``*`` operator.
        """

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.y)
        result_with_C = self.C.matvec_transp(self.y)

        for j in range(self.ncol):
            self.assertTrue(result_with_A[j] == result_with_C[j])



# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_htransp``.
        """

        result_with_A = self.A.matvec_htransp(self.y)
        result_with_C = self.C.matvec_htransp(self.y)

        for j in range(self.ncol):
            self.assertTrue(result_with_A[j] == result_with_C[j])



##################################
# Case Symmetric, Non Zero
##################################
class CySparseCommonNumpyVectorMultiplication_Symmetric_CSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX128_T, itype=INT32_T, store_symmetry=True)

        # numpy vectors
        self.x = np.empty(self.size, dtype=np.complex128)


        self.x.fill(2 + 2j)



        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x

        for i in range(self.size):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.x)
        result_with_C = self.C.matvec_transp(self.x)

        for i in range(self.size):
            self.assertTrue(result_with_A[i] == result_with_C[i])



##################################
# Case Non Symmetric, Zero
##################################
class CySparseCommonNumpyVectorMultiplication_WithZeroCSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX128_T, itype=INT32_T, store_zero=True)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.complex128)
        self.y = np.empty(self.nrow, dtype=np.complex128)

        self.x.fill(2 + 2j)
        self.y.fill(2 + 2j)



        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.y)
        result_with_C = self.C.matvec_transp(self.y)

        for j in range(self.ncol):
            self.assertTrue(result_with_A[j] == result_with_C[j])



##################################
# Case Symmetric, Zero
##################################
class CySparseCommonNumpyVectorMultiplication_Symmetric_WithZero_CSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX128_T, itype=INT32_T, store_symmetry=True, store_zero=True)

        # numpy vectors
        self.x = np.empty(self.size, dtype=np.complex128)


        self.x.fill(2 + 2j)



        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x

        for i in range(self.size):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.x)
        result_with_C = self.C.matvec_transp(self.x)

        for i in range(self.size):
            self.assertTrue(result_with_A[i] == result_with_C[i])



########################################################################################################################
# WITH STRIDES IN THE NUMPY VECTORS
########################################################################################################################
##################################
# Case Non Symmetric, Non Zero
##################################
class CySparseCommonNumpyVectorWithStrideMultiplication_CSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.stride_factor = 3

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX128_T, itype=INT32_T)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.complex128)
        self.x_strided = np.empty(self.ncol * self.stride_factor, dtype=np.complex128)
        self.y = np.empty(self.nrow, dtype=np.complex128)
        self.y_strided = np.empty(self.nrow * self.stride_factor, dtype=np.complex128)

        self.x.fill(2 + 2j)
        self.x_strided.fill(1+1j)
        self.y.fill(2 + 2j)
        self.y_strided.fill(1+1j)

        # make both strided and non strided vectors alike:
        for j in range(self.ncol):
            self.x_strided[j * self.stride_factor] = self.x[j]

        for i in range(self.nrow):
            self.y_strided[i * self.stride_factor] = self.y[i]


        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x_strided[::self.stride_factor]

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.y)
        result_with_C = self.C.matvec_transp(self.y_strided[::self.stride_factor])

        for j in range(self.ncol):
            self.assertTrue(result_with_A[j] == result_with_C[j])



##################################
# Case Symmetric, Non Zero
##################################
class CySparseCommonNumpyVectorWithStrideMultiplication_Symmetric_CSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.stride_factor = 3

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX128_T, itype=INT32_T, store_symmetry=True)

        # numpy vectors
        self.x = np.empty(self.size, dtype=np.complex128)
        self.x_strided = np.empty(self.size * self.stride_factor, dtype=np.complex128)

        self.x.fill(2 + 2j)
        self.x_strided.fill(1+1j)

        # make both strided and non strided vectors alike:
        for j in range(self.size):
            self.x_strided[j * self.stride_factor] = self.x[j]


        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x_strided[::self.stride_factor]

        for i in range(self.size):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.x)
        result_with_C = self.C.matvec_transp(self.x_strided[::self.stride_factor])

        for j in range(self.size):
            self.assertTrue(result_with_A[j] == result_with_C[j])




##################################
# Case Non Symmetric, Zero
##################################
class CySparseCommonNumpyVectorWithStrideMultiplication_WithZeroCSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.stride_factor = 3

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX128_T, itype=INT32_T, store_zero=True)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.complex128)
        self.x_strided = np.empty(self.ncol * self.stride_factor, dtype=np.complex128)
        self.y = np.empty(self.nrow, dtype=np.complex128)
        self.y_strided = np.empty(self.nrow * self.stride_factor, dtype=np.complex128)

        self.x.fill(2 + 2j)
        self.x_strided.fill(1+1j)
        self.y.fill(2 + 2j)
        self.y_strided.fill(1+1j)

        # make both strided and non strided vectors alike:
        for j in range(self.ncol):
            self.x_strided[j * self.stride_factor] = self.x[j]

        for i in range(self.nrow):
            self.y_strided[i * self.stride_factor] = self.y[i]



        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x_strided[::self.stride_factor]

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.y)
        result_with_C = self.C.matvec_transp(self.y_strided[::self.stride_factor])

        for j in range(self.ncol):
            self.assertTrue(result_with_A[j] == result_with_C[j])



##################################
# Case Symmetric, Zero
##################################
class CySparseCommonNumpyVectorWithStrideMultiplication_Symmetric_WithZero_CSRSparseMatrix_INT32_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.stride_factor = 3

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX128_T, itype=INT32_T, store_symmetry=True, store_zero=True)

        # numpy vectors
        self.x = np.empty(self.size, dtype=np.complex128)
        self.x_strided = np.empty(self.size * self.stride_factor, dtype=np.complex128)

        self.x.fill(2 + 2j)
        self.x_strided.fill(1+1j)

        # make both strided and non strided vectors alike:
        for j in range(self.size):
            self.x_strided[j * self.stride_factor] = self.x[j]


        self.C = self.A.to_csr()


# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x_strided[::self.stride_factor]

        for i in range(self.size):
            self.assertTrue(result_with_A[i] == result_with_C[i])


# ======================================================================================================================
    def test_numpy_vector_matvec_transp_element_by_element(self):
        """
        Test ``matvec_transp``.
        """

        result_with_A = self.A.matvec_transp(self.x)
        result_with_C = self.C.matvec_transp(self.x_strided[::self.stride_factor])

        for j in range(self.size):
            self.assertTrue(result_with_A[j] == result_with_C[j])



if __name__ == '__main__':
    unittest.main()