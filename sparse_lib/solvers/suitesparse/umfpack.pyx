from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

import sys

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

    int umfpack_di_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, void * numeric)

    int umfpack_di_get_numeric(int * Lp, int * Lj, double * Lx, double * Lz,
                               int * Up, int * Ui, double * Ux, double * Uz,
                               int * P, int * Q, double * Dx, double * Dz,
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

UMFPACK_SYS_LIST = [
        'UMFPACK_A',
        'UMFPACK_At',
        'UMFPACK_Aat',
        'UMFPACK_Pt_L',
        'UMFPACK_L',
        'UMFPACK_Lt_P',
        'UMFPACK_Lat_P',
        'UMFPACK_Lt',
        'UMFPACK_U_Qt',
        'UMFPACK_U',
        'UMFPACK_Q_Ut',
        'UMFPACK_Q_Uat',
        'UMFPACK_Ut',
        'UMFPACK_Uat'
    ]

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
            test_umfpack_result(status, "create_symbolic")

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
            test_umfpack_result(status, "create_numeric")


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

