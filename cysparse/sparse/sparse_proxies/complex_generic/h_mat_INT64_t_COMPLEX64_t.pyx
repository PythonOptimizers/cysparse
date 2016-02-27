#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.sparse.s_mat cimport SparseMatrix, MakeMatrixLikeString

from cysparse.common_types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type
from cysparse.sparse.s_mat cimport PyLLSparseMatrix_Check

from cysparse.sparse.operator_proxies.mul_proxy import MulProxy
from cysparse.sparse.operator_proxies.sum_proxy import SumProxy
from cysparse.sparse.operator_proxies.scalar_mul_proxy import ScalarMulProxy

from cysparse.common_types.cysparse_types import is_scalar

cimport numpy as cnp

cnp.import_array()

from cpython cimport Py_INCREF, Py_DECREF, PyObject
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

cdef class ConjugateTransposedSparseMatrix_INT64_t_COMPLEX64_t:
    """
    Proxy to the conjugate transposed matrix of a :class:`SparseMatrix`.

    """
    ####################################################################################################################
    # Init and properties
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
    def nnz(self):
        return self.A.nnz

    
    @property
    def dtype(self):
        return self.A.cp_type.dtype

    
    @property
    def itype(self):
        return self.A.cp_type.itype

    
    @property
    def is_symmetric(self):
        return self.A.is_symmetric

    
    @property
    def store_symmetric(self):
        return self.A.store_symmetric

    
    @property
    def store_zero(self):
        return self.A.store_zero

    
    @property
    def is_mutable(self):
        return self.A.__is_mutable

    
    @property
    def base_type_str(self):
        return 'Conjugate Transposed of ' + self.A.base_type_str

    
    @property
    def full_type_str(self):
        return 'Conjugate Transposed of ' + self.A.full_type_str

    
    @property
    def itype_str(self):
        return self.A.itype_str

    
    @property
    def dtype_str(self):
        return self.A.dtype_str

    
    @property
    def nargin(self):
        return self.A.nargout

    
    @property
    def nargout(self):
        return self.A.nargin

    # for compatibility with numpy, PyKrylov, etc
    
    @property
    def shape(self):
        return self.A.ncol, self.A.nrow

    def __dealloc__(self):
        Py_DECREF(self.A) # release ref

    def __repr__(self):
        return "Proxy to the conjugate transposed (.H) of %s" % self.A

    ####################################################################################################################
    # End of Common code
    ####################################################################################################################
    
    @property
    def H(self):
        return self.A

    
    @property
    def T(self):
        return self.A.conj

    
    @property
    def conj(self):
        return self.A.T

    ####################################################################################################################
    # Set/get
    ####################################################################################################################
    # EXPLICIT TYPE TESTS
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a transposed matrix")

        return conjf(self.A[key[1], key[0]])


    ####################################################################################################################
    # Basic operations
    ####################################################################################################################
    def __mul__(self, B):
        if cnp.PyArray_Check(B) and B.ndim == 1:
            return self.matvec(B)

        elif is_scalar(B):
            return ScalarMulProxy(B, self)

        elif is_scalar(self):
            return ScalarMulProxy(self, B)

        return MulProxy(self, B)

    def __add__(self, B):
        """
        Return a :class:`SumProxy`.

        Returns:
            A :class:`SumProxy`, i.e. a proxy to a matrix-like sum.

        """
        if is_scalar(B) or is_scalar(self):
            raise RuntimeError("This operation is not allowed")

        return SumProxy(self, B)

    def __sub__(self, B):
        """
        Return a :class:`SumProxy`.

        Returns:
            A :class:`SumProxy`, i.e. a proxy to a matrix-like sum.

        """
        if is_scalar(B) or is_scalar(self):
            raise RuntimeError("This operation is not allowed")

        return SumProxy(self, B, real_sum=False)

    def matvec(self, B):
        return self.A.matvec_adj(B)

    def matvec_transp(self, B):
        return self.A.matvec_conj(B)

    def matvec_adj(self, B):
        return self.A.matvec(B)

    def matvec_conj(self, B):
        return self.A.matvec_transp(B)

    def copy(self):
        raise NotImplementedError('This proxy is unique')

    def matrix_copy(self):
        return self.A.create_adjoint()

    def matdot(self, B):
        raise NotImplementedError('Operation not implemented yet')

    def matdot_transp(self, B):
        raise NotImplementedError('Operation not implemented yet')

    ####################################################################################################################
    # String representations
    ####################################################################################################################
    def __repr__(self):
        """
        Return an unique representation of the :class:`ConjugateTransposedSparseMatrix` object.

        """
        return "Proxy to the conjugate transposed (.H) of %s" % self.A.__repr__()

    def _matrix_description_before_printing(self):
        return "Proxy to the conjugate transposed (.H) of %s" % self.A._matrix_description_before_printing()

    def at_to_string(self, i, j, int cell_width=10):
        return self.A.at_conj_to_string(j, i, cell_width)

    def __str__(self):
        """
        Return a string to print the :class:`SparseMatrix` object to screen.

        """
        s = self._matrix_description_before_printing()
        s += '\n'
        s += MakeMatrixLikeString(self)

        return s