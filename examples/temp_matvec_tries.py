from cysparse.sparse.ll_mat import *
from scipy.sparse import lil_matrix

import random as rd
import numpy as np


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









if __name__ == "__main__":
    rd.seed()

    nbr_elements = 80
    size = 8000

    A = LLSparseMatrix(size=size, size_hint=nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
    B = LLSparseMatrix(size=size, size_hint=nbr_elements, itype=INT32_T, dtype=FLOAT64_T)

    A_s = lil_matrix((size, size), dtype=np.float64)
    B_s = lil_matrix((size, size), dtype=np.float64)


    matrix_list = []
    matrix_list.append(A)
    matrix_list.append(B)
    matrix_list.append(A_s)
    matrix_list.append(B_s)

    construct_random_matrices(matrix_list, size, nbr_elements)

    print A
    print B

    v = np.arange(0, size, dtype=np.float64)

    w  = A * B * v

    print w

    print "=" * 80

    w_s = A_s * B_s * v

    assert np.array_equal(w, w_s)


