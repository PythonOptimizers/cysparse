from cysparse.sparse.sparse_mat cimport SparseMatrix

cimport numpy as cnp

cnp.import_array()

from python_ref cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)


cdef class TransposedSparseMatrix:

    def __cinit__(self, SparseMatrix A):
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

    property T:
        def __get__(self):
            return self.A

        def __set__(self, value):
            raise AttributeError('Attribute T (transposed) is read-only')

        def __del__(self):
            raise AttributeError('Attribute T (transposed) is read-only')

    def __dealloc__(self):
        Py_DECREF(self.A) # release ref

    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a transposed matrix")

        return self.A[key[1], key[0]]

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

    def matvec(self, B):
        return self.A.matvec_transp(B)

    def matvec_transp(self, B):
        return self.A.matvec(B)

    def matrix_copy(self):
        #return self.A.create_submatrix()
        pass

