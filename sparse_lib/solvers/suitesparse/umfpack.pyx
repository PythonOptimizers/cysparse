from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

cdef extern from "suitesparse/umfpack.h":

    char * UMFPACK_DATE

    # OPAQUE UMFPACK OBJECTS
    cdef struct Symbolic:
        pass
    ctypedef Symbolic * symbolic_t

    cdef struct Numeric:
        pass
    ctypedef Numeric * numeric_t

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
                            symbolic_t * symbolic,
                            double * control, double * info)

    int umfpack_di_numeric(int * Ap, int * Ai, double * Ax,
                           symbolic_t symbolic,
                           numeric_t * numeric,
                           double * control, double * info)

    void umfpack_di_free_symbolic(symbolic_t * symbolic)
    void umfpack_di_free_numeric(numeric_t * numeric)
    void umfpack_di_defaults(double * control)

    int umfpack_di_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, numeric_t numeric)

    int umfpack_di_get_numeric(int * Lp, int * Lj, double * Lx, double * Lz,
                               int * Up, int * Ui, double * Ux, double * Uz,
                               int * P, int * Q, double * Dx, double * Dz,
                               int * do_recip, double * Rs,
                               numeric_t numeric)


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
    UMFPACK_VERSION = "%s.%s.%s (%s)" % (UMFPACK_MAIN_VERSION,
                                     UMFPACK_SUB_VERSION,
                                     UMFPACK_SUBSUB_VERSION,
                                     UMFPACK_DATE)
    def __cinit__(self, LLSparseMatrix A):
        self.A = A
        self.nrow = A.nrow
        self.ncol = A.ncol

        self.csc  = self.A.to_csc()

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
    cdef create_symbolic(self, recompute=False):

        if self.symbolic_computed:
            if not recompute:
                pass
            else:
                self.free_symbolic()

        #cdef double* info = <double>self.info.data
        #cdef double * control = self.control.data
        #cdef symbolic_t symbolic = self.symbolic



        #status= umfpack_di_symbolic(self.nrow, self.ncol, ind, col, val, &self.symbolic, &self.control, &self.info)

        #if status != UMFPACK_OK:
        #    self.free_symbolic()
        #    test_umfpack_result(status, "create_symbolic()")
        #else:
        #   self.symbolic_computed = True

    # def create_numeric(self, recompute=False):
    #
    #     if self.numeric is not None:
    #         if not recompute:
    #             pass
    #         else:
    #             self.free_numeric()
    #
    #     self.create_symbolic()
    #
    #     status, self.numeric = self.UMFPACK_ROUTINES['numeric'](self.ind, self.col, self.val, self.symbolic, self.control, self.info)
    #
    #     if status != UMFPACK_OK:
    #         self.numeric = None
    #         test_umfpack_result(status, "create_numeric()")


