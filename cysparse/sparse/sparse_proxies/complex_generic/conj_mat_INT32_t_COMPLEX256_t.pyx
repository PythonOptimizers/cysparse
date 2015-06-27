from cysparse.sparse.s_mat cimport SparseMatrix

from cysparse.types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type
from cysparse.sparse.ll_mat cimport PyLLSparseMatrix_Check

cimport numpy as cnp

cnp.import_array()

from python_ref cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)

    double creal(double complex z)
    double cimag(double complex z)

    long double creall(long double complex z)
    long double cimagl(long double complex z)

    double cabs(double complex z)
    float cabsf(float complex z)
    long double cabsl(long double complex z)

    double complex conj(double complex z)
    float complex  conjf (float complex z)
    long double complex conjl (long double complex z)

cdef class ConjugatedSparseMatrix_INT32_t_COMPLEX256_t:
    """
    Proxy to the conjugated matrix of a :class:`SparseMatrix`.

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

    property nrow:
        def __get__(self):
            return self.A.nrow

        def __set__(self, value):
            raise AttributeError('Attribute nrow is read-only')

        def __del__(self):
            raise AttributeError('Attribute nrow is read-only')

    property ncol:
        def __get__(self):
            return self.A.ncol

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
            return self.A.nrow, self.A.ncol

        def __set__(self, value):
            raise AttributeError('Attribute shape is read-only')

        def __del__(self):
            raise AttributeError('Attribute shape is read-only')

    def __dealloc__(self):
        Py_DECREF(self.A) # release ref

    def __repr__(self):
        return "Proxy to the conjugated (.conj) of %s" % self.A

    ####################################################################################################################
    # End of Common code
    ####################################################################################################################
    
    @property
    def H(self):
        return self.A.T

    
    @property
    def T(self):
        return self.A.H

    
    @property
    def conj(self):
        return self.A

    ####################################################################################################################
    # Set/get
    ####################################################################################################################
    # EXPLICIT TYPE TESTS
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a transposed matrix")

        return conjl(self.A[key[0], key[1]])


    ####################################################################################################################
    # Basic operations
    ####################################################################################################################
    def __mul__(self, B):
        raise NotImplementedError("Multiplication with this kind of object not implemented yet...")

    def matvec(self, B):
        return self.A.matvec_conj(B)

    def matvec_transp(self, B):
        return self.matvec_htransp(B)

    def matvec_htransp(self, B):
        return self.A.matvec_transp(B)

    def matvec_conj(self, B):
        return self.A.matvec(B)

    def copy(self):
        raise NotImplementedError('This proxy is unique')

    def matrix_copy(self):
        return self.A.create_conjugate()