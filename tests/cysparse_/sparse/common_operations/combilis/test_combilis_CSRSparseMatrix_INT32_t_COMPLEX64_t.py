#!/usr/bin/env python

"""
This file tests XXX for all matrix-likes objects.

"""

import unittest
from cysparse.sparse.ll_mat import *


########################################################################################################################
# Helpers
########################################################################################################################
def are_matrices_equal(A, B):
    real_A = A
    real_B = B

    try:
        real_A = A.to_ll()
    except:
        pass

    try:
        real_B = B.to_ll()
    except:
        pass

    A_nrow, A_ncol = real_A.shape
    B_nrow, B_ncol = real_B.shape

    if A_nrow != B_nrow or A_ncol != B_ncol:
        return False

    for i in xrange(A_nrow):
        for j in xrange(A_ncol):
            if real_A[i, j] != real_B[i, j]:
                return False

    return True


########################################################################################################################
# Tests
########################################################################################################################

NROW = 10
NCOL = 14
SIZE = 10


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparsecombilisNoSymmetryNoZero_CSRSparseMatrix_INT32_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX64_T, itype=INT32_T)

        # numpy vectors

        self.x = np.empty(self.ncol, dtype=np.complex64)



        self.x.fill(2 + 2j)



        self.C = self.A.to_csr()



    ### Equality element by element

    def test_simple_addition(self):
        self.assertTrue(are_matrices_equal(self.C + self.C, 2 * self.C))

    def test_simple_substraction(self):
        self.assertTrue(are_matrices_equal(self.C + self.C - self.C, self.C))

    def test_complicated_addition(self):
        self.assertTrue(are_matrices_equal(2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj, 2 * self.C - 3 * self.C.conj))

    def test_simple_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(4 * self.C, self.C * 4))

    def test_complicated_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(2 * self.C * 3, 6 * self.C))

    ### Equality of multiplication with a NumPy vector

    def test_simple_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C) * self.x, (2 * self.C) * self.x)

    def test_simple_substraction_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C - self.C) * self.x, self.C * self.x)

    def test_complicated_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj) * self.x, (2 * self.C - 3 * self.C.conj) * self.x)

    def test_simple_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((4 * self.C) * self.x, (self.C * 4) * self.x)

    def test_complicated_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C * 3) * self.x, (6 * self.C) * self.x)

#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparsecombilisWithSymmetryNoZero_CSRSparseMatrix_INT32_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX64_T, itype=INT32_T, store_symmetry=True)

        # numpy vectors
        self.x = np.empty(self.size, dtype=np.complex64)

        self.x.fill(2 + 2j)



        self.C = self.A.to_csr()



    ### Equality element by element

    def test_simple_addition(self):
        self.assertTrue(are_matrices_equal(self.C + self.C, 2 * self.C))

    def test_simple_substraction(self):
        self.assertTrue(are_matrices_equal(self.C + self.C - self.C, self.C))

    def test_complicated_addition(self):
        self.assertTrue(are_matrices_equal(2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj, 2 * self.C - 3 * self.C.conj))

    def test_simple_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(4 * self.C, self.C * 4))

    def test_complicated_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(2 * self.C * 3, 6 * self.C))

    ### Equality of multiplication with a NumPy vector

    def test_simple_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C) * self.x, (2 * self.C) * self.x)

    def test_simple_substraction_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C - self.C) * self.x, self.C * self.x)

    def test_complicated_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj) * self.x, (2 * self.C - 3 * self.C.conj) * self.x)

    def test_simple_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((4 * self.C) * self.x, (self.C * 4) * self.x)

    def test_complicated_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C * 3) * self.x, (6 * self.C) * self.x)


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparsecombilisNoSymmetrySWithZero_CSRSparseMatrix_INT32_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX64_T, itype=INT32_T, store_zero=True)

        # numpy vectors

        self.x = np.empty(self.ncol, dtype=np.complex64)



        self.x.fill(2 + 2j)



        self.C = self.A.to_csr()



    ### Equality element by element

    def test_simple_addition(self):
        self.assertTrue(are_matrices_equal(self.C + self.C, 2 * self.C))

    def test_simple_substraction(self):
        self.assertTrue(are_matrices_equal(self.C + self.C - self.C, self.C))

    def test_complicated_addition(self):
        self.assertTrue(are_matrices_equal(2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj, 2 * self.C - 3 * self.C.conj))

    def test_simple_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(4 * self.C, self.C * 4))

    def test_complicated_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(2 * self.C * 3, 6 * self.C))

    ### Equality of multiplication with a NumPy vector

    def test_simple_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C) * self.x, (2 * self.C) * self.x)

    def test_simple_substraction_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C - self.C) * self.x, self.C * self.x)

    def test_complicated_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj) * self.x, (2 * self.C - 3 * self.C.conj) * self.x)

    def test_simple_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((4 * self.C) * self.x, (self.C * 4) * self.x)

    def test_complicated_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C * 3) * self.x, (6 * self.C) * self.x)


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparsecombilisWithSymmetrySWithZero_CSRSparseMatrix_INT32_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX64_T, itype=INT32_T, store_symmetry=True, store_zero=True)

        # numpy vectors
        self.x = np.empty(self.size, dtype=np.complex64)

        self.x.fill(2 + 2j)



        self.C = self.A.to_csr()



    ### Equality element by element

    def test_simple_addition(self):
        self.assertTrue(are_matrices_equal(self.C + self.C, 2 * self.C))

    def test_simple_substraction(self):
        self.assertTrue(are_matrices_equal(self.C + self.C - self.C, self.C))

    def test_complicated_addition(self):
        self.assertTrue(are_matrices_equal(2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj, 2 * self.C - 3 * self.C.conj))

    def test_simple_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(4 * self.C, self.C * 4))

    def test_complicated_commutative_scalar_multiplication(self):
        self.assertTrue(are_matrices_equal(2 * self.C * 3, 6 * self.C))

    ### Equality of multiplication with a NumPy vector

    def test_simple_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C) * self.x, (2 * self.C) * self.x)

    def test_simple_substraction_with_a_numpy_vector(self):
        np.testing.assert_array_equal((self.C + self.C - self.C) * self.x, self.C * self.x)

    def test_complicated_addition_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C + self.C - self.C - 2 * self.C.conj - self.C.conj) * self.x, (2 * self.C - 3 * self.C.conj) * self.x)

    def test_simple_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((4 * self.C) * self.x, (self.C * 4) * self.x)

    def test_complicated_commutative_scalar_multiplication_with_a_numpy_vector(self):
        np.testing.assert_array_equal((2 * self.C * 3) * self.x, (6 * self.C) * self.x)


if __name__ == '__main__':
    unittest.main()
