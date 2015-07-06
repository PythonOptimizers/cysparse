from cysparse.types.cysparse_types cimport *
from cysparse.types.cysparse_types import *

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT = 40        # allocated size by default

unexposed_value = object()


cdef __set_store_zeros_attribute(SparseMatrix A, bint store_zeros):
    """
    Access private  ``__store_zeros`` attribute and change it.

    Args:
        A: ``SparseMatrix``.
        store_zeros: Boolean value to set the attribute to.

    """
    A.__store_zeros = store_zeros

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
        self.__store_zeros = False

    def __enter__(self):
        self.__store_zeros = self.A.store_zeros
        __set_store_zeros_attribute(self.A, False)

    def __exit__(self, type, value, traceback):
        __set_store_zeros_attribute(self.A, self.__store_zeros)

cpdef bint PySparseMatrix_Check(object obj):
    """
    Test if ``obj`` is a ``SparseMatrix`` or not.

    Args:
        obj: Whatever.

    Return:
        ``True`` if ``obj`` is a ``SparseMatrix`` object or inherited from it.
    """
    return isinstance(obj, SparseMatrix)

########################################################################################################################
# BASE MATRIX CLASS
########################################################################################################################
cdef class SparseMatrix:

    def __cinit__(self, **kwargs):
        """

        Warning:
            Only use named arguments! This is on purpose!
        """
        assert unexposed_value == kwargs.get('control_object', None), "Matrix must be instantiated with a factory method"

        self.__type_name = "Not defined"
        self.__type = "Not defined"
        self.__index_and_type = "[not defined, not defined]"

        # This is overwritten in inheriting classes
        #self.cp_type.itype = kwargs.get('itype', INT32_T)
        #self.cp_type.dtype = kwargs.get('dtype', FLOAT64_T)

        self.__is_symmetric = kwargs.get('is_symmetric', False)
        self.__store_zeros = kwargs.get('store_zeros', False)
        self.__is_mutable = False

    # for compatibility with numpy, PyKrylov, etc
    @property
    def shape(self):
        return (self.nrow, self.ncol)

    @property
    def dtype(self):
        return self.cp_type.dtype

    @property
    def itype(self):
        return self.cp_type.itype

    @property
    def is_symmetric(self):
        return self.__is_symmetric

    @property
    def is_mutable(self):
        return self.__is_mutable

    @property
    def store_zeros(self):
        return self.__store_zeros

    @property
    def type(self):
        return self.__type

    @property
    def type_name(self):
        return self.__type_name

    ####################################################################################################################
    # Basic common methods
    ####################################################################################################################
    # All methods raise NotImplementedError. We could have provided a common method that would have been refined in
    # the respective children classes but we preferred to force an optimized version for all classes inheriting from
    # this class, i.e. if it works, it is optimized for that particular class, if not, it must be implemented if needed.

    #########################
    # Sub matrices
    #########################
    # Copy
    def copy(self):
        """
        Return a **deep** copy of itself.

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def diag(self, n = 0):
        """
        Return diagonal in a :program:`NumPy` array.

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def diags(self, diag_coeff):
        """
        Return a list of diagonals with coefficients in ``diag_coeff``.

        Args:
            diag_coeff: List or slice of diagonals coefficients.
        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")

    def to_ndarray(self):
        """
        Return the matrix in the form of a :program:`NumPy` ``ndarray``.

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
    # Internal arrays
    #########################
    def get_c_pointers(self):
        """
        Return C pointers to internal arrays.

        """
        raise NotImplementedError("Operation not allowed or not implemented.")

    def get_numpy_arrays(self):
        """
        Return :program:`NumPy` arrays equivalent to internal C-arrays.
        """
        raise NotImplementedError("Operation not allowed or not implemented.")

    #########################
    # Printing
    #########################
    def to_string(self, **kwargs):
        """
        Return a string representing the **content** of matrix.

        Args:
            kwargs: Arguments are named.

        """
        raise NotImplementedError("Operation not implemented (yet). Please report.")


cdef MakeMatrixString(object A, full=False):
    """
    Return a print of the :class:`SparseMatrix` object.

    Args:
        A: A matrix like object (:class:`SparseMatrix` or :class:`ProxySparseMatrix`).
        full: If ``True`` overwrite settings and print the **full** matrix.

    Note:
        This is the **only** function to print matrix like objects.

    """
    # This the main and only function to print matrix objects.
    # TODO: add better coordination between this function and 'element_to_string_@type@'. The latter uses some widths
    # (9, 6 and 2) that correspond to the widths defined here...
    cdef:
        Py_ssize_t MAX_MATRIX_HEIGHT = 11
        Py_ssize_t MAX_MATRIX_WIDTH = 11
        Py_ssize_t cell_width = 10

        Py_ssize_t max_height, max_width, i, j, frontier

    s = ''
    empty_cell = "...".center(cell_width + 1)

    if is_complex_type(A.dtype):
        empty_cell = empty_cell + empty_cell
        MAX_MATRIX_WIDTH = MAX_MATRIX_WIDTH / 2

    if not full and (A.nrow > MAX_MATRIX_HEIGHT or A.ncol > MAX_MATRIX_WIDTH):
        max_height = min(A.nrow, MAX_MATRIX_HEIGHT)
        max_width  = min(A.ncol, MAX_MATRIX_WIDTH)

        for i from 0 <= i < max_height:
            frontier = max_width - i
            for j from 0 <= j < max_width:
                if j < frontier:
                    s += "%s " % A.at_to_string(i, j, cell_width)
                elif j == frontier:
                    s += empty_cell
                else:
                    s += "%s " % A.at_to_string(A.nrow - max_height + i, A.ncol - max_width + j, cell_width)
            s += '\n'
        s += '\n'
    else:  # full matrix
        for i from 0 <= i < A.nrow:
            for j from 0 <= j < A.ncol:
                s += "%s " % A.at_to_string(i, j, cell_width)
            s += '\n'
        s += '\n'

    return s