from sparse_lib.sparse.ll_mat cimport LLSparseMatrix
from sparse_lib.sparse.csr_mat cimport CSRSparseMatrix, MakeCSRSparseMatrix
from sparse_lib.sparse.csc_mat cimport CSCSparseMatrix, MakeCSCSparseMatrix


from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "umfpack.h":

    char * UMFPACK_DATE

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
        self.A = A
        self.nrow = A.nrow
        self.ncol = A.ncol

        assert self.nrow == self.ncol, "Only square matrices are handled in UMFPACK"

        # TODO: type csc in pxd file
        self.csc_mat  = self.A.to_csc()

        #cdef double * val = self.val
        #cdef int * col = <int *> self.col
        #cdef int * ind = <int *> self.ind

        self.symbolic_computed = False
        self.numeric_computed = False

        # set default parameters for control
        umfpack_di_defaults(<double *>&self.control)
        self.control[UMFPACK_PRL] = 3

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        self.free()

    def free_symbolic(self):
        if self.symbolic_computed:
            umfpack_di_free_symbolic(&self.symbolic)

    def free_numeric(self):
        if self.numeric_computed:
            umfpack_di_free_numeric(&self.numeric)

    def free(self):
        self.free_numeric()
        self.free_symbolic()

    ####################################################################################################################
    # PRIMARY ROUTINES
    ####################################################################################################################
    cdef int _create_symbolic(self):

        if self.symbolic_computed:
            self.free_symbolic()

        cdef int * ind = <int *> self.csc_mat.ind
        cdef int * row = <int *> self.csc_mat.row
        cdef double * val = <double *> self.csc_mat.val


        cdef int status= umfpack_di_symbolic(self.nrow, self.ncol, ind, row, val, &self.symbolic, self.control, self.info)

        self.symbolic_computed = True

        return status


    def create_symbolic(self, recompute=False):
        if not recompute and self.symbolic_computed:
            return

        cdef int status = self._create_symbolic()

        if status != UMFPACK_OK:
            self.free_symbolic()
            test_umfpack_result(status, "create_symbolic()")

    cdef int _create_numeric(self):

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

        cdef int status =  umfpack_di_solve(UMFPACK_SYS_DICT[umfpack_sys], ind, row, val, <double*> sol.data, <double *> b.data, self.numeric, self.control, self.info)

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

    def get_LU(self):
        """
        Return LU factorisation objects. If needed, the LU factorisation is triggered.

        Returns:
            (L, U, P, Q, D, do_recip, R)

        """
        # TODO: use properties?? we can only get matrices, not set them...
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

        cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] P
        cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] Q
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] D
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] R

        cdef cnp.npy_intp *dims_n_row = [n_row]
        cdef cnp.npy_intp *dims_n_col = [n_col]
        #cdef cnp.ndarray[cnp.int_t, ndim=1] result = \
        cdef cnp.npy_intp *dims_min = [min(n_row, n_col)]

        P = cnp.PyArray_EMPTY(1, dims_n_row, cnp.NPY_INTP, 0)
        Q = cnp.PyArray_EMPTY(1, dims_n_col, cnp.NPY_INTP, 0)

        D = cnp.PyArray_EMPTY(1, dims_min, cnp .NPY_DOUBLE, 0)

        R = cnp.PyArray_EMPTY(1, dims_n_row, cnp .NPY_DOUBLE, 0)


        cdef int status =umfpack_di_get_numeric(Lp, Lj, Lx,
                               Up, Ui, Ux,
                               <int *> P.data, <int *> Q.data, <double *> D.data,
                               &_do_recip, <double *> R.data,
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
        """
        self.control[UMFPACK_PRL] = level

    def get_verbosity(self):
        """
        Return UMFPACK verbosity level.

        Returns:
            verbosity_level (int): The verbosity level set.
        """
        return self.control[UMFPACK_PRL]

    def report_control(self):
        """
        Print control values.
        """
        umfpack_di_report_control(self.control)

    def report_info(self):
         """
         Print all status information.

         Use **after** calling :meth:`create_symbolic()`, :meth:`create_numeric()`, :meth:`factorize()` or :meth:`solve()`.
         """
         umfpack_di_report_info(self.control, self.info)

    def report_symbolic(self):
         """
         Print information about the opaque ``symbolic`` object.
         """
         if not self.symbolic_computed:
             print "No opaque symbolic object has been computed"
             return

         umfpack_di_report_symbolic(self.symbolic, self.control)

    def report_numeric(self):
         """
         Print information about the opaque ``numeric`` object.
         """
         umfpack_di_report_numeric(self.numeric, self.control)

