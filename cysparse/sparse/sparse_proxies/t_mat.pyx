# TODO: verify if we need to generate this file
# For the moment I (Nikolaj) 'm leaving it as it is just in case things change...

from cysparse.types.cysparse_types import *
from cysparse.sparse.s_mat cimport SparseMatrix, MakeMatrixString
#from cysparse.sparse.sparse_proxies cimport ProxySparseMatrix

from cysparse.types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type
from cysparse.sparse.ll_mat cimport PyLLSparseMatrix_Check

cimport numpy as cnp

cnp.import_array()

from python_ref cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)


cdef class TransposedSparseMatrix:
    """
    Proxy to the transposed matrix of a :class:`SparseMatrix`.

    """
    ####################################################################################################################
    # Init and properties
    ####################################################################################################################
    ####################################################################################################################
    # Common code from p_mat.pyx See #113: I could not solve the circular dependencies...
    ####################################################################################################################
    def __cinit__(self, SparseMatrix A):
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

    def get_matrix(self):
        """
        Return pointer to original matrix ``A``.
        """
        return self.A

    
    @property
    def nrow(self):
        return self.A.ncol

    
    @property
    def ncol(self):
        return self.A.nrow

    
    @property
    def dtype(self):
        return self.A.cp_type.dtype

    
    @property
    def itype(self):
        return self.A.cp_type.itype

    # for compatibility with numpy, PyKrylov, etc
    
    @property
    def shape(self):
        return (self.nrow, self.ncol)

    def __dealloc__(self):
        Py_DECREF(self.A) # release ref

    ####################################################################################################################
    # End of Common code
    ####################################################################################################################
    
    @property
    def T(self):
        return self.A

    
    @property
    def H(self):
        return self.A.conj

    
    @property
    def conj(self):
        return self.A.H

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
        # This call is needed as ``__mul__`` doesn't find self.A ...
        return self._mul(B)

    def _mul(self, B):
        """

        """
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

    def matvec_htransp(self, B):
        if not is_complex_type(self.A.cp_type.dtype):
            raise TypeError("This operation is only valid for complex matrices")
        return self.A.matvec_conj(B)

    def matvec_conj(self, B):
        if not is_complex_type(self.A.cp_type.dtype):
            raise TypeError("This operation is only valid for complex matrices")
        return self.A.matvec_htransp(B)

    def copy(self):
        raise NotImplementedError('This proxy is unique')

    def matrix_copy(self):
        return self.A.create_transpose()

    def print_to(self, OUT):
        return self.A.print_to(OUT, transposed=True)

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def __repr__(self):
        """
        Return an unique representation of the :class:`TransposedSparseMatrix` object.

        """
        return "Proxy to the transposed (.T) of %s" % self.A.__repr__()

    def _matrix_description_before_printing(self):
        return "Proxy to the transposed (.T) of %s" % self.A._matrix_description_before_printing()

    def at_to_string(self, i, j, int cell_width=10):
        return self.A.at_to_string(j, i, cell_width)

    def __str__(self):
        """
        Return a string to print the :class:`SparseMatrix` object to screen.

        """
        s = self._matrix_description_before_printing()
        s += '\n'
        s += MakeMatrixString(self)

        return s