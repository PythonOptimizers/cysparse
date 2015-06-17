from cysparse.sparse.s_mat import SparseMatrix

from cysparse.types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type
from cysparse.sparse.ll_mat cimport PyLLSparseMatrix_Check

cimport numpy as cnp

cnp.import_array()

from python_ref cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)


#cdef class TransposedSparseMatrix(ProxySparseMatrix):
cdef class TransposedSparseMatrix:
    """
    Proxy to the transposed matrix of a :class:`SparseMatrix`.

    """
    ####################################################################################################################
    # Init and properties
    ####################################################################################################################
    def __cinit__(self):
        pass

    property T:
        def __get__(self):
            return self.A

        def __set__(self, value):
            raise AttributeError('Attribute T (transposed) is read-only')

        def __del__(self):
            raise AttributeError('Attribute T (transposed) is read-only')

    def __dealloc__(self):
        pass

    ####################################################################################################################
    # Set/get
    ####################################################################################################################
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a transposed matrix")

        return self.A[key[1], key[0]]

    ####################################################################################################################
    # Basic operations
    ####################################################################################################################
    def __mul__(self, B):
        if cnp.PyArray_Check(B):
            # test type
            assert are_mixed_types_compatible(self.dtype, B.dtype), "Multiplication only allowed with a Numpy compatible type (%s)!" % cysparse_to_numpy_type(self.dtype)

            if B.ndim == 2:
                return self.A.matdot(B)
            elif B.ndim == 1:
                return self.A.matvec_transp(B)
            else:
                raise IndexError("Matrix dimensions must agree")
        elif PyLLSparseMatrix_Check(B):
            return self.A.matdot(B)
        else:
            raise NotImplementedError("Multiplication with this kind of object not implemented yet...")

    def matvec(self, B):
        return self.A.matvec_transp(B)

    def matvec_transp(self, B):
        return self.A.matvec(B)

    def matrix_copy(self):
        #return self.A.create_submatrix()
        pass

    def print_to(self, OUT):
        return self.A.print_to(OUT, transposed=True)
