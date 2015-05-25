from cysparse.sparse.sparse_mat cimport SparseMatrix

cimport numpy as cnp

cnp.import_array()

cdef class TransposedSparseMatrix:

    def __cinit__(self, SparseMatrix A):
        self.A = A

    def __mul__(self, B):
        if cnp.PyArray_Check(B):
            # test type
            # TODO
            #assert are_mixed_types_compatible(@type|type2enum@, B.dtype), "Multiplication only allowed with a Numpy compatible type (%s)!" % cysparse_to_numpy_type(@type|type2enum@)

            if B.ndim == 2:
                #return multiply_ll_mat_with_numpy_ndarray(self, B)
                raise NotImplementedError("Multiplication with this kind of object not implemented yet...")
            elif B.ndim == 1:
                return self.A.matvec_transp(B)
            else:
                raise IndexError("Matrix dimensions must agree")
        else:
            raise NotImplementedError("Multiplication with this kind of object not implemented yet...")