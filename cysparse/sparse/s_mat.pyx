from cysparse.types.cysparse_types cimport *
from cysparse.types.cysparse_types import *

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default

unexposed_value = object()


class NonZeros():
    """
    Context manager to use methods with flag ``store_zeros`` to ``False``.

    The initial value is restored upon completion.

    Use like this:

        >>>with NonZeros(A):
        >>>    ...
    """
    def __init__(self, SparseMatrix A):
        self.A = A
        self.store_zeros = False

    def __enter__(self):
        self.store_zeros = self.A.store_zeros
        self.A.store_zeros = False

    def __exit__(self, type, value, traceback):
        self.A.store_zeros = self.store_zeros

cpdef bint PySparseMatrix_Check(object obj):
    return isinstance(obj, SparseMatrix)

########################################################################################################################
# BASE MATRIX CLASS
########################################################################################################################
cdef class SparseMatrix:

    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments!
        """
        assert unexposed_value == kwargs.get('control_object', None), "Matrix must be instantiated with a factory method"

        self.type_name = "Not defined"
        self.cp_type.itype = kwargs.get('itype', INT32_T)
        self.cp_type.dtype = kwargs.get('dtype', FLOAT64_T)

        self.is_symmetric = kwargs.get('is_symmetric', False)
        self.store_zeros = kwargs.get('store_zeros', False)
        self.is_mutable = False

    # for compatibility with numpy, PyKrylov, etc
    property shape:
        def __get__(self):
            return (self.nrow, self.ncol)

        def __set__(self, value):
            raise AttributeError('Attribute shape is read-only')

        def __del__(self):
            raise AttributeError('Attribute shape is read-only')

    property dtype:
        def __get__(self):
            return self.cp_type.dtype

        def __set__(self, value):
            raise AttributeError('Attribute dtype is read-only')

        def __del__(self):
            raise AttributeError('Attribute dtype is read-only')

    property itype:
        def __get__(self):
            return self.cp_type.itype

        def __set__(self, value):
            raise AttributeError('Attribute itype is read-only')

        def __del__(self):
            raise AttributeError('Attribute itype is read-only')

    ####################################################################################################################
    # Basic common methods
    ####################################################################################################################
    #########################
    # Sub matrices
    #########################
    def diag(self):
        """
        Return diagonal in a :program:`NumPy` array.

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    #########################
    # Multiplication with vectors
    #########################
    def matvec(self, B):
        """
        Return ``A * B`` with ``B`` a :program:`NumPy` vector.

        Args:
            B: A :program:`NumPy` vector.

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def matvec_transp(self, B):
        """
        Return ``A^t * B`` with ``B`` a :program:`NumPy` vector.

        Args:
            B: A :program:`NumPy` vector.

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def matvec_htransp(self, B):
        """
        Return ``A^h * B`` with ``B`` a :program:`NumPy` vector.

        Args:
            B: A :program:`NumPy` vector.

        """
        if not is_complex_type(self.cp_type.dtype):
            raise TypeError("This operation is only valid for complex matrices")

        raise NotImplementedError("Operation not implemented (yet). Please report.")


    def matvec_conj(self, B):
        """
        Return ``conj(A) * B`` with ``B`` a :program:`NumPy` vector.

        Args:
            B: A :program:`NumPy` vector.

        """
        if not is_complex_type(self.cp_type.dtype):
            raise TypeError("This operation is only valid for complex matrices")

        raise NotImplementedError("Operation not implemented (yet). Please report.")

    #########################
    # Multiplication with 2d matrices
    #########################
    def matdot(self, B):
        """
        Return ``A * B``.

        Args:
            B:

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def matdot_transp(self, B):
        """
        Return ``A^t * B``.

        Args:
            B:

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def matdot_htransp(self, B):
        """
        Return ``A^h * B``.

        Args:
            B:

        """
        if not is_complex_type(self.cp_type.dtype):
            raise TypeError("This operation is only valid for complex matrices")

        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def matdot_conj(self, B):
        """
        Return ``conj(A) * B``.

        Args:
            B:

        """
        if not is_complex_type(self.cp_type.dtype):
            raise TypeError("This operation is only valid for complex matrices")

        raise NotImplementedError("Operation not implemented (yet). Please report.")

    #########################
    # Memory
    #########################
    # Copy
    def copy(self):
        """
        Return a **deep** copy of itself.

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

