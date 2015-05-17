from cysparse.types.cysparse_types cimport *

assert FLOAT_T == FLOAT64_T, "UMFPACK only deals with double precision (FLOAT64)"

from cysparse.sparse.ll_mat cimport LLSparseMatrix
from cysparse.sparse.csr_mat cimport CSRSparseMatrix, MakeCSRSparseMatrix
from cysparse.sparse.csc_mat cimport CSCSparseMatrix, MakeCSCSparseMatrix

import numpy as np
cimport numpy as cnp


cnp.import_array()


cdef extern from "umfpack.h":

    char * UMFPACK_DATE
    ctypedef long SuiteSparse_long

    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO

        UMFPACK_VERSION, UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION

        # Return codes:
        UMFPACK_OK

        UMFPACK_WARNING_singular_matrix, UMFPACK_WARNING_determinant_underflow
        UMFPACK_WARNING_determinant_overflow

        UMFPACK_ERROR_out_of_memory
        UMFPACK_ERROR_invalid_Numeric_object
        UMFPACK_ERROR_invalid_Symbolic_object
        UMFPACK_ERROR_argument_missing
        UMFPACK_ERROR_n_nonpositive
        UMFPACK_ERROR_invalid_matrix
        UMFPACK_ERROR_different_pattern
        UMFPACK_ERROR_invalid_system
        UMFPACK_ERROR_invalid_permutation
        UMFPACK_ERROR_internal_error
        UMFPACK_ERROR_file_IO

        # Control:
        # Printing routines:
        UMFPACK_PRL
        # umfpack_*_symbolic:
        UMFPACK_DENSE_ROW
        UMFPACK_DENSE_COL
        UMFPACK_BLOCK_SIZE
        UMFPACK_STRATEGY
        UMFPACK_2BY2_TOLERANCE
        UMFPACK_FIXQ
        UMFPACK_AMD_DENSE
        UMFPACK_AGGRESSIVE
        # umfpack_*_numeric:
        UMFPACK_PIVOT_TOLERANCE
        UMFPACK_ALLOC_INIT
        UMFPACK_SYM_PIVOT_TOLERANCE
        UMFPACK_SCALE
        UMFPACK_FRONT_ALLOC_INIT
        UMFPACK_DROPTOL
        # umfpack_*_solve:
        UMFPACK_IRSTEP

        # For UMFPACK_STRATEGY:
        UMFPACK_STRATEGY_AUTO
        UMFPACK_STRATEGY_UNSYMMETRIC
        UMFPACK_STRATEGY_2BY2
        UMFPACK_STRATEGY_SYMMETRIC

        # For UMFPACK_SCALE:
        UMFPACK_SCALE_NONE
        UMFPACK_SCALE_SUM
        UMFPACK_SCALE_MAX

        # for SOLVE ACTIONS
        UMFPACK_A
        UMFPACK_At
        UMFPACK_Aat
        UMFPACK_Pt_L
        UMFPACK_L
        UMFPACK_Lt_P
        UMFPACK_Lat_P
        UMFPACK_Lt
        UMFPACK_U_Qt
        UMFPACK_U
        UMFPACK_Q_Ut
        UMFPACK_Q_Uat
        UMFPACK_Ut
        UMFPACK_Uat

    ####################################################################################################################
    # DI VERSION:  real double precision, int integers
    ####################################################################################################################
    int umfpack_di_symbolic(int n_row, int n_col,
                            int * Ap, int * Ai, double * Ax,
                            void ** symbolic,
                            double * control, double * info)

    int umfpack_di_numeric(int * Ap, int * Ai, double * Ax,
                           void * symbolic,
                           void ** numeric,
                           double * control, double * info)

    void umfpack_di_free_symbolic(void ** symbolic)
    void umfpack_di_free_numeric(void ** numeric)
    void umfpack_di_defaults(double * control)

    int umfpack_di_solve(int umfpack_sys, int * Ap, int * Ai, double * Ax, double * x, double * b, void * numeric, double * control, double * info)

    int umfpack_di_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, void * numeric)

    int umfpack_di_get_numeric(int * Lp, int * Lj, double * Lx,
                               int * Up, int * Ui, double * Ux,
                               int * P, int * Q, double * Dx,
                               int * do_recip, double * Rs,
                               void * numeric)

    void umfpack_di_report_control(double *)
    void umfpack_di_report_info(double *, double *)
    void umfpack_di_report_symbolic(void *, double *)
    void umfpack_di_report_numeric(void *, double *)

    ####################################################################################################################
    # DL VERSION:   real double precision, SuiteSparse long integers
    ####################################################################################################################
    SuiteSparse_long umfpack_dl_symbolic(SuiteSparse_long n_row, SuiteSparse_long n_col,
                            SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax,
                            void ** symbolic,
                            double * control, double * info)

    SuiteSparse_long umfpack_dl_numeric(SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax,
                           void * symbolic,
                           void ** numeric,
                           double * control, double * info)

    void umfpack_dl_free_symbolic(void ** symbolic)
    void umfpack_dl_free_numeric(void ** numeric)
    void umfpack_dl_defaults(double * control)

    SuiteSparse_long umfpack_dl_solve(SuiteSparse_long umfpack_sys, SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax, double * x, double * b, void * numeric, double * control, double * info)

    SuiteSparse_long umfpack_dl_get_lunz(SuiteSparse_long * lnz, SuiteSparse_long * unz, SuiteSparse_long * n_row, SuiteSparse_long * n_col,
                            SuiteSparse_long * nz_udiag, void * numeric)

    SuiteSparse_long umfpack_dl_get_numeric(SuiteSparse_long * Lp, SuiteSparse_long * Lj, double * Lx,
                               SuiteSparse_long * Up, SuiteSparse_long * Ui, double * Ux,
                               SuiteSparse_long * P, SuiteSparse_long * Q, double * Dx,
                               SuiteSparse_long * do_recip, double * Rs,
                               void * numeric)

    void umfpack_dl_report_control(double *)
    void umfpack_dl_report_info(double *, double *)
    void umfpack_dl_report_symbolic(void *, double *)
    void umfpack_dl_report_numeric(void *, double *)

    ####################################################################################################################
    # ZI VERSION:   complex double precision, int integers
    ####################################################################################################################
    int umfpack_zi_symbolic(int n_row, int n_col,
                            int * Ap, int * Ai, double * Ax, double * Az,
                            void ** symbolic,
                            double * control, double * info)

    int umfpack_zi_numeric(int * Ap, int * Ai, double * Ax, double * Az,
                           void * symbolic,
                           void ** numeric,
                           double * control, double * info)

    void umfpack_zi_free_symbolic(void ** symbolic)
    void umfpack_zi_free_numeric(void ** numeric)
    void umfpack_zi_defaults(double * control)

    int umfpack_zi_solve(int umfpack_sys, int * Ap, int * Ai, double * Ax,  double * Az, double * x, double * b, void * numeric, double * control, double * info)

    int umfpack_zi_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, void * numeric)

    int umfpack_zi_get_numeric(int * Lp, int * Lj, double * Lx,
                               int * Up, int * Ui, double * Ux,
                               int * P, int * Q, double * Dx,
                               int * do_recip, double * Rs,
                               void * numeric)

    void umfpack_zi_report_control(double *)
    void umfpack_zi_report_info(double *, double *)
    void umfpack_zi_report_symbolic(void *, double *)
    void umfpack_zi_report_numeric(void *, double *)

    ####################################################################################################################
    # ZL VERSION:   complex double precision, SuiteSparse long integers
    ####################################################################################################################
    SuiteSparse_long umfpack_zl_symbolic(SuiteSparse_long n_row, SuiteSparse_long n_col,
                            SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax, double * Az,
                            void ** symbolic,
                            double * control, double * info)

    SuiteSparse_long umfpack_zl_numeric(SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax, double * Az,
                           void * symbolic,
                           void ** numeric,
                           double * control, double * info)

    void umfpack_zl_free_symbolic(void ** symbolic)
    void umfpack_zl_free_numeric(void ** numeric)
    void umfpack_zl_defaults(double * control)

    SuiteSparse_long umfpack_zl_solve(SuiteSparse_long umfpack_sys, SuiteSparse_long * Ap, SuiteSparse_long * Ai, double * Ax,  double * Az, double * x, double * b, void * numeric, double * control, double * info)

    SuiteSparse_long umfpack_zl_get_lunz(SuiteSparse_long * lnz, SuiteSparse_long * unz, SuiteSparse_long * n_row, SuiteSparse_long * n_col,
                            SuiteSparse_long * nz_udiag, void * numeric)

    SuiteSparse_long umfpack_zl_get_numeric(SuiteSparse_long * Lp, SuiteSparse_long * Lj, double * Lx,
                               SuiteSparse_long * Up, SuiteSparse_long * Ui, double * Ux,
                               SuiteSparse_long * P, SuiteSparse_long * Q, double * Dx,
                               SuiteSparse_long * do_recip, double * Rs,
                               void * numeric)

    void umfpack_zl_report_control(double *)
    void umfpack_zl_report_info(double *, double *)
    void umfpack_zl_report_symbolic(void *, double *)
    void umfpack_zl_report_numeric(void *, double *)

def umfpack_version():
    version_string = "UMFPACK version %s" % UMFPACK_VERSION

    return version_string

def umfpack_detailed_version():
    version_string = "%s.%s.%s (%s)" % (UMFPACK_MAIN_VERSION,
                                         UMFPACK_SUB_VERSION,
                                         UMFPACK_SUBSUB_VERSION,
                                         UMFPACK_DATE)
    return version_string

UMFPACK_SYS_DICT = {
        'UMFPACK_A'     : UMFPACK_A,
        'UMFPACK_At'    : UMFPACK_At,
        'UMFPACK_Aat'   : UMFPACK_Aat,
        'UMFPACK_Pt_L'  : UMFPACK_Pt_L,
        'UMFPACK_L'     : UMFPACK_L,
        'UMFPACK_Lt_P'  : UMFPACK_Lt_P,
        'UMFPACK_Lat_P' : UMFPACK_Lat_P,
        'UMFPACK_Lt'    : UMFPACK_Lt,
        'UMFPACK_U_Qt'  : UMFPACK_U_Qt,
        'UMFPACK_U'     : UMFPACK_U,
        'UMFPACK_Q_Ut'  : UMFPACK_Q_Ut,
        'UMFPACK_Q_Uat' : UMFPACK_Q_Uat,
        'UMFPACK_Ut'    : UMFPACK_Ut,
        'UMFPACK_Uat'   : UMFPACK_Uat
    }

UMFPACK_ERROR_CODE_DICT = {
        UMFPACK_OK: 'UMFPACK_OK',
        UMFPACK_WARNING_singular_matrix: 'UMFPACK_WARNING_singular_matrix',
        UMFPACK_WARNING_determinant_underflow: 'UMFPACK_WARNING_determinant_underflow',
        UMFPACK_WARNING_determinant_overflow: 'UMFPACK_WARNING_determinant_overflow',
        UMFPACK_ERROR_out_of_memory: 'UMFPACK_ERROR_out_of_memory',
        UMFPACK_ERROR_invalid_Numeric_object: 'UMFPACK_ERROR_invalid_Numeric_object',
        UMFPACK_ERROR_invalid_Symbolic_object: 'UMFPACK_ERROR_invalid_Symbolic_object',
        UMFPACK_ERROR_argument_missing: 'UMFPACK_ERROR_argument_missing',
        UMFPACK_ERROR_n_nonpositive: 'UMFPACK_ERROR_n_nonpositive',
        UMFPACK_ERROR_invalid_matrix: 'UMFPACK_ERROR_invalid_matrix',
        UMFPACK_ERROR_different_pattern: 'UMFPACK_ERROR_different_pattern',
        UMFPACK_ERROR_invalid_system: 'UMFPACK_ERROR_invalid_system',
        UMFPACK_ERROR_invalid_permutation: 'UMFPACK_ERROR_invalid_permutation',
        UMFPACK_ERROR_internal_error: 'UMFPACK_ERROR_internal_error',
        UMFPACK_ERROR_file_IO: 'UMFPACK_ERROR_file_IO'
}

def test_umfpack_result(status, msg, raise_error=True, print_on_screen=True):
    """
    Test returned status from UMFPACK routines.

    Args:
        status (int): Returned status from UMFPACK routines.
        msg: Message to display in error or on screen.
        raise_error: Raises an error if ``status`` is an error if ``True``..
        print_on_screen: Prints warnings on screen if ``True``.

    Raises:
        RuntimeError: If ``raise_error`` is ``True`` and ``status < 0``.

    """

    if status != UMFPACK_OK:

        if status < 0 and raise_error:
            raise RuntimeError("%s %s: %s" % (msg, "aborted", UMFPACK_ERROR_CODE_DICT[status]))
        elif status > 0 and print_on_screen:
            print "%s %s: %s" % (msg, "warning", UMFPACK_ERROR_CODE_DICT[status])


cdef class UmfpackSolver:
    """
    Umfpack Solver from SuiteSparse.

    This version **only** deals with ``LLSparseMatrix`` objects.

    We follow the common use of Umfpack. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Umfpack.
    """
    UMFPACK_VERSION = "%s.%s.%s (%s)" % (UMFPACK_MAIN_VERSION,
                                     UMFPACK_SUB_VERSION,
                                     UMFPACK_SUBSUB_VERSION,
                                     UMFPACK_DATE)


    def __cinit__(self, LLSparseMatrix A):
        """
        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        self.A = A
        self.nrow = A.nrow
        self.ncol = A.ncol

        # test if we can use UMFPACK
        assert self.nrow == self.ncol, "Only square matrices are handled in UMFPACK"



        self.is_complex = A.is_complex

        # fix UMFPACK family: 'di', 'dl', 'zi' or 'zl'
        if self.is_complex:
            self.family = 'z'
        else:
            self.family = 'd'

        if INT_T == INT64_T:
            self.family += 'l'
        elif INT_T == INT32_T:
            self.family += 'i'
        else:
            raise TypeError("UMFPACK only works with INT32 or INT64 for matrix indices")

        # TODO: implement both cases!
        if INT_T == INT64_T:
            raise NotImplemented("UMFPACK library not (yet) interfaced with INT64 (long)")

        if A.is_complex:
            raise NotImplemented("Complex version of UMFPACK not (yet) implemented")

        self.csc_mat  = self.A.to_csc()

        self.symbolic_computed = False
        self.numeric_computed = False

        # set default parameters for control
        umfpack_di_defaults(<double *>&self.control)
        self.set_verbosity(3)

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """
        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        self.free()

    def free_symbolic(self):
        """
        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        if self.symbolic_computed:
            umfpack_di_free_symbolic(&self.symbolic)

    def free_numeric(self):
        """
        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        if self.numeric_computed:
            umfpack_di_free_numeric(&self.numeric)

    def free(self):
        """
        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        self.free_numeric()
        self.free_symbolic()

    ####################################################################################################################
    # PRIMARY ROUTINES
    ####################################################################################################################
    cdef int _create_symbolic(self):
        """
        {{COMPLEX: NO}}
        {{GENERIC: NO}}
        """

        if self.symbolic_computed:
            self.free_symbolic()

        cdef INT_t * ind = <INT_t *> self.csc_mat.ind
        cdef INT_t * row = <INT_t *> self.csc_mat.row
        cdef double * val = <double *> self.csc_mat.val
        cdef double * ival

        cdef int status

        if self.is_complex:
            # TODO: add complex type for CSC
            #ival = <double *> self.csc_mat.ival
            status= umfpack_zi_symbolic(self.nrow, self.ncol, ind, row, val, ival, &self.symbolic, self.control, self.info)
        else:
            status= umfpack_di_symbolic(self.nrow, self.ncol, ind, row, val, &self.symbolic, self.control, self.info)

        self.symbolic_computed = True

        return status


    def create_symbolic(self, recompute=False):
        """
        {{COMPLEX: NO}}
        {{GENERIC: NO}}
        """
        if not recompute and self.symbolic_computed:
            return

        cdef int status = self._create_symbolic()

        if status != UMFPACK_OK:
            self.free_symbolic()
            test_umfpack_result(status, "create_symbolic()")

    cdef int _create_numeric(self):
        """
        {{COMPLEX: NO}}
        {{GENERIC: NO}}
        """

        if self.numeric_computed:
            self.free_numeric()

        cdef int * ind = <int *> self.csc_mat.ind
        cdef int * row = <int *> self.csc_mat.row
        cdef double * val = <double *> self.csc_mat.val

        cdef int status =  umfpack_di_numeric(ind, row, val,
                           self.symbolic,
                           &self.numeric,
                           self.control, self.info)

        self.numeric_computed = True

        return status

    def create_numeric(self, recompute=False):
        """
        {{COMPLEX: NO}}
        {{GENERIC: NO}}
        """

        if not recompute and self.numeric_computed:
            return

        cdef int status = self._create_numeric()

        if status != UMFPACK_OK:
            self.free_numeric()
            test_umfpack_result(status, "create_numeric()")


    def solve(self, cnp.ndarray[cnp.double_t, ndim=1, mode="c"] b, umfpack_sys='UMFPACK_A', irsteps=2):
        """
        Solve the linear system  ``A x = b``.

        Args:
           b: a Numpy vector of appropriate dimension.
           umfpack_sys: specifies the type of system being solved:

                    +-------------------+--------------------------------------+
                    |``"UMFPACK_A"``    | :math:`\mathbf{A} x = b` (default)   |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_At"``   | :math:`\mathbf{A}^T x = b`           |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Pt_L"`` | :math:`\mathbf{P}^T \mathbf{L} x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_L"``    | :math:`\mathbf{L} x = b`             |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Lt_P"`` | :math:`\mathbf{L}^T \mathbf{P} x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Lt"``   | :math:`\mathbf{L}^T x = b`           |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_U_Qt"`` | :math:`\mathbf{U} \mathbf{Q}^T x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_U"``    | :math:`\mathbf{U} x = b`             |
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Q_Ut"`` | :math:`\mathbf{Q} \mathbf{U}^T x = b`|
                    +-------------------+--------------------------------------+
                    |``"UMFPACK_Ut"``   | :math:`\mathbf{U}^T x = b`           |
                    +-------------------+--------------------------------------+

           irsteps: number of iterative refinement steps to attempt. Default: 2

        Returns:
            ``sol``: The solution of ``A*x=b`` if everything went well.

        Raises:
            AttributeError: When vector ``b`` doesn't have a ``shape`` attribute.
            AssertionError: When vector ``b`` doesn't have the right first dimension.
            RuntimeError: Whenever ``UMFPACK`` returned status is not ``UMFPACK_OK`` and is an error.

        Notes:
            The opaque objects ``symbolic`` and ``numeric`` are automatically created if necessary.

            You can ask for a report of what happened by calling :meth:`report_info()`.

        {{COMPLEX: NO}}
        {{GENERIC: NO}}
        """
        #TODO: add other umfpack_sys arguments to the docstring.
        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")
        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        if umfpack_sys not in UMFPACK_SYS_DICT.keys():
            raise ValueError('umfpack_sys must be in' % UMFPACK_SYS_DICT.keys())

        self.control[UMFPACK_IRSTEP] = irsteps

        self.create_symbolic()
        self.create_numeric()

        cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.double)

        cdef int * ind = <int *> self.csc_mat.ind
        cdef int * row = <int *> self.csc_mat.row
        cdef double * val = <double *> self.csc_mat.val

        cdef int status =  umfpack_di_solve(UMFPACK_SYS_DICT[umfpack_sys], ind, row, val, <double*> cnp.PyArray_DATA(sol), <double *> cnp.PyArray_DATA(b), self.numeric, self.control, self.info)

        if status != UMFPACK_OK:
            test_umfpack_result(status, "solve()")

        return sol

    ####################################################################################################################
    # LU ROUTINES
    ####################################################################################################################
    def get_lunz(self):
        """
        Determine the size and number of non zeros in the LU factors held by the opaque ``Numeric`` object.

        Returns:
            (lnz, unz, n_row, n_col, nz_udiag):

            lnz: The number of nonzeros in ``L``, including the diagonal (which is all one's)
            unz: The number of nonzeros in ``U``, including the diagonal.
            n_row, n_col: The order of the ``L`` and ``U`` matrices. ``L`` is ``n_row`` -by- ``min(n_row,n_col)``
                and ``U`` is ``min(n_row,n_col)`` -by- ``n_col``.
            nz_udiag: The number of numerically nonzero values on the diagonal of ``U``. The
                matrix is singular if ``nz_diag < min(n_row,n_col)``. A ``divide-by-zero``
                will occur if ``nz_diag < n_row == n_col`` when solving a sparse system
                involving the matrix ``U`` in ``solve()``.

        Raises:
            RuntimeError: When ``UMFPACK`` return status is not ``UMFPACK_OK`` and is an error.

        {{COMPLEX: NO}}
        {{GENERIC: NO}}
        """
        self.create_numeric()

        cdef:
            int lnz
            int unz
            int n_row
            int n_col
            int nz_udiag

        cdef status = umfpack_di_get_lunz(&lnz, &unz, &n_row, &n_col, &nz_udiag, self.numeric)

        if status != UMFPACK_OK:
            test_umfpack_result(status, "get_lunz()")

        return (lnz, unz, n_row, n_col, nz_udiag)

    def get_LU(self, get_L=True, get_U=True, get_P=True, get_Q=True, get_D=True, get_R=True):
        """
        Return LU factorisation objects. If needed, the LU factorisation is triered.

        Returns:
            (L, U, P, Q, D, do_recip, R)

            The original matrix A is factorized into

                L U = P R A Q

            where:
             - L is unit lower triangular,
             - U is upper triangular,
             - P and Q are permutation matrices,
             - R is a row-scaling diagonal matrix such that

                  * the i-th row of A has been multiplied by R[i] if do_recip = True,
                  * the i-th row of A has been divided by R[i] if do_recip = False.

            L and U are returned as CSRSparseMatrix and CSCSparseMatrix sparse matrices respectively.
            P, Q and R are returned as NumPy arrays.


        {{COMPLEX: NO}}
        {{GENERIC: NO}}
        """
        # TODO: use properties?? we can only get matrices, not set them...
        # TODO: implement the use of L=True, U=True, P=True, Q=True, D=True, R=True
        # i.e. allow to return only parts of the arguments and not necessarily all of them...
        self.create_numeric()

        cdef:
            int lnz
            int unz
            int n_row
            int n_col
            int nz_udiag

            int _do_recip

        (lnz, unz, n_row, n_col, nz_udiag) = self.get_lunz()

        # L CSR matrix
        cdef int * Lp = <int *> PyMem_Malloc((n_row + 1) * sizeof(int))
        if not Lp:
            raise MemoryError()

        cdef int * Lj = <int *> PyMem_Malloc(lnz * sizeof(int))
        if not Lj:
            raise MemoryError()

        cdef double * Lx = <double *> PyMem_Malloc(lnz * sizeof(double))
        if not Lx:
            raise MemoryError()

        # U CSC matrix
        cdef int * Up = <int *> PyMem_Malloc((n_col + 1) * sizeof(int))
        if not Up:
            raise MemoryError()

        cdef int * Ui = <int *> PyMem_Malloc(unz * sizeof(int))
        if not Ui:
            raise MemoryError()

        cdef double * Ux = <double *> PyMem_Malloc(unz * sizeof(double))
        if not Ux:
            raise MemoryError()

        # TODO: see what type of int exactly to pass
        cdef cnp.npy_intp *dims_n_row = [n_row]
        cdef cnp.npy_intp *dims_n_col = [n_col]

        cdef cnp.npy_intp *dims_min = [min(n_row, n_col)]

        #cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] P
        cdef cnp.ndarray[int, ndim=1, mode='c'] P

        P = cnp.PyArray_EMPTY(1, dims_n_row, cnp.NPY_INT32, 0)

        #cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] Q
        cdef cnp.ndarray[int, ndim=1, mode='c'] Q
        Q = cnp.PyArray_EMPTY(1, dims_n_col, cnp.NPY_INT32, 0)

        cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] D
        D = cnp.PyArray_EMPTY(1, dims_min, cnp.NPY_DOUBLE, 0)

        cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] R
        R = cnp.PyArray_EMPTY(1, dims_n_row, cnp.NPY_DOUBLE, 0)



        cdef int status =umfpack_di_get_numeric(Lp, Lj, Lx,
                               Up, Ui, Ux,
                               <int *> cnp.PyArray_DATA(P), <int *> cnp.PyArray_DATA(Q), <double *> cnp.PyArray_DATA(D),
                               &_do_recip, <double *> cnp.PyArray_DATA(R),
                               self.numeric)

        if status != UMFPACK_OK:
            test_umfpack_result(status, "get_LU()")

        cdef bint do_recip = _do_recip

        cdef CSRSparseMatrix L
        cdef CSCSparseMatrix U

        cdef int size = min(n_row,n_col)
        L = MakeCSRSparseMatrix(nrow=size, ncol=size, nnz=lnz, ind=Lp, col=Lj, val=Lx)
        U = MakeCSCSparseMatrix(nrow=size, ncol=size, nnz=unz, ind=Up, row=Ui, val=Ux)

        return (L, U, P, Q, D, do_recip, R)

    ####################################################################################################################
    # REPORTING ROUTINES
    ####################################################################################################################
    def set_verbosity(self, level):
        """
        Set UMFPACK verbosity level.

        Args:
            level (int): Verbosity level (default: 1).


        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        self.control[UMFPACK_PRL] = level

    def get_verbosity(self):
        """
        Return UMFPACK verbosity level.

        Returns:
            verbosity_level (int): The verbosity level set.

        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        return self.control[UMFPACK_PRL]

    def report_control(self):
        """
        Print control values.

        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        if self.is_complex:
            if INT_T == INT64_T:
                umfpack_zl_report_control(self.control)
            else:
                umfpack_zi_report_control(self.control)
        else:
            if INT_T == INT64_T:
                umfpack_dl_report_control(self.control)
            else:
                umfpack_di_report_control(self.control)

    def report_info(self):
        """
        Print all status information.

        Use **after** calling :meth:`create_symbolic()`, :meth:`create_numeric()`, :meth:`factorize()` or :meth:`solve()`.

        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        if self.is_complex:
            if INT_T == INT64_T:
                umfpack_zl_report_info(self.control, self.info)
            else:
                umfpack_zi_report_info(self.control, self.info)
        else:
            if INT_T == INT64_T:
                umfpack_dl_report_info(self.control, self.info)
            else:
                umfpack_di_report_info(self.control, self.info)

    def report_symbolic(self):
        """
        Print information about the opaque ``symbolic`` object.

        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        if not self.symbolic_computed:
            print "No opaque symbolic object has been computed"
            return

        if self.is_complex:
            if INT_T == INT64_T:
                umfpack_zl_report_symbolic(self.symbolic, self.control)
            else:
                umfpack_zi_report_symbolic(self.symbolic, self.control)
        else:
            if INT_T == INT64_T:
                umfpack_dl_report_symbolic(self.symbolic, self.control)
            else:
                umfpack_di_report_symbolic(self.symbolic, self.control)

    def report_numeric(self):
        """
        Print information about the opaque ``numeric`` object.

        {{COMPLEX: YES}}
        {{GENERIC: YES}}
        """
        if self.is_complex:
            if INT_T == INT64_T:
                umfpack_zl_report_numeric(self.numeric, self.control)
            else:
                umfpack_zi_report_numeric(self.numeric, self.control)
        else:
            if INT_T == INT64_T:
                umfpack_dl_report_numeric(self.numeric, self.control)
            else:
                umfpack_di_report_numeric(self.numeric, self.control)

