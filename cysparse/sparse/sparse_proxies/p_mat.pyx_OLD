from cysparse.sparse.s_mat cimport SparseMatrix
from cysparse.types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type
from cysparse.sparse.ll_mat cimport PyLLSparseMatrix_Check

cimport numpy as cnp

cnp.import_array()

from cpython cimport Py_DECREF, Py_INCREF, PyObject

cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)


cdef class ProxySparseMatrix:
    """
    Proxy to a :class:`SparseMatrix` object.

    """
    ####################################################################################################################
    # Init and properties
    ####################################################################################################################
    def __cinit__(self, SparseMatrix A):
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly


    property nrow:
        def __get__(self):
            return self.A.ncol

        def __set__(self, value):
            raise AttributeError('Attribute nrow is read-only')

        def __del__(self):
            raise AttributeError('Attribute nrow is read-only')

    property ncol:
        def __get__(self):
            return self.A.nrow

        def __set__(self, value):
            raise AttributeError('Attribute ncol is read-only')

        def __del__(self):
            raise AttributeError('Attribute ncol is read-only')

    property dtype:
        def __get__(self):
            return self.A.cp_type.dtype

        def __set__(self, value):
            raise AttributeError('Attribute dtype is read-only')

        def __del__(self):
            raise AttributeError('Attribute dtype is read-only')

    property itype:
        def __get__(self):
            return self.A.cp_type.itype

        def __set__(self, value):
            raise AttributeError('Attribute itype is read-only')

        def __del__(self):
            raise AttributeError('Attribute itype is read-only')

    # for compatibility with numpy, PyKrylov, etc
    property shape:
        def __get__(self):
            return self.A.ncol, self.A.nrow

        def __set__(self, value):
            raise AttributeError('Attribute shape is read-only')

        def __del__(self):
            raise AttributeError('Attribute shape is read-only')

    def __dealloc__(self):
        Py_DECREF(self.A) # release ref


