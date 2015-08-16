from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF


from cysparse.types.cysparse_generic_types cimport split_array_complex_values_kernel_INT64_t_COMPLEX128_t, join_array_complex_values_kernel_INT64_t_COMPLEX128_t


cdef extern from "cholmod.h":

    char * CHOLMOD_DATE
    #ctypedef long SuiteSparse_long # doesn't work... why?
    cdef enum:
        CHOLMOD_MAIN_VERSION
        CHOLMOD_SUB_VERSION
        CHOLMOD_SUBSUB_VERSION
        CHOLMOD_VERSION


    cdef enum:
        # Five objects
        CHOLMOD_COMMON
        CHOLMOD_SPARSE
        CHOLMOD_FACTOR
        CHOLMOD_DENSE
        CHOLMOD_TRIPLET

    # we only use REAL and ZOMPLEX
    cdef enum:
        CHOLMOD_PATTERN  	# pattern only, no numerical values
        CHOLMOD_REAL		# a real matrix
        CHOLMOD_COMPLEX     # a complex matrix (ANSI C99 compatible)
        CHOLMOD_ZOMPLEX     # a complex matrix (MATLAB compatible)

    # itype: we only use INT and LONG
    cdef enum:
        CHOLMOD_INT         # all integer arrays are int
        CHOLMOD_INTLONG     # most are int, some are SuiteSparse_long
        CHOLMOD_LONG        # all integer arrays are SuiteSparse_long

    # dtype: float or double
    cdef enum:
        CHOLMOD_DOUBLE      # all numerical values are double
        CHOLMOD_SINGLE

    int cholmod_l_start(cholmod_common *Common)
    int cholmod_l_finish(cholmod_common *Common)

    int cholmod_l_defaults(cholmod_common *Common)

    # Common struct
    int cholmod_l_check_common(cholmod_common *Common)
    int cholmod_l_print_common(const char *name, cholmod_common *Common)

    # Sparse struct
    int cholmod_l_check_sparse(cholmod_sparse *A, cholmod_common *Common)
    int cholmod_l_print_sparse(cholmod_sparse *A, const char *name, cholmod_common *Common)

    # Factor struct
    int cholmod_l_check_factor(cholmod_factor *L, cholmod_common *Common)
    int cholmod_l_print_factor(cholmod_factor *L, const char *name, cholmod_common *Common)

    # Triplet struct
    #int cholmod_l_check_triplet(cholmod_triplet *T, cholmod_common *Common)
    


########################################################################################################################
# CHOLMOD HELPERS
########################################################################################################################
# Populating a sparse matrix in CHOLMOD is done in two times:
# - first (populate1), we give the common attributes and
# - second (populate2), we split the values array in two if needed (complex case) and give the values (real or complex).

cdef populate1_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct, CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat, bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t` matrix.

    First part: common attributes for both real and complex matrices.

    Note:
        We only use the ``cholmod_sparse`` **packed** version.
    """
    assert no_copy, "The version with copy is not implemented yet..."

    sparse_struct.nrow = csc_mat.nrow
    sparse_struct.ncol = csc_mat.ncol
    sparse_struct.nzmax = csc_mat.nnz

    sparse_struct.p = csc_mat.ind
    sparse_struct.i = csc_mat.row

    # TODO: change this when we'll accept symmetric matrices **without** symmetric storage scheme
    sparse_struct.stype = -1

    # itype: can be CHOLMOD_INT or CHOLMOD_LONG: we don't use the mixed version CHOLMOD_INTLONG

    sparse_struct.itype = CHOLMOD_LONG



    #sparse_struct.dtype


cdef populate2_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct,
                                                              CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat,
                                                              FLOAT64_t * csc_mat_rval,
                                                              FLOAT64_t * csc_mat_ival,
                                                              bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t` matrix.

    Second part: Non common attributes for complex matrices.

    Note:
        We only use the ``cholmod_sparse`` **packed** version.
    """
    assert no_copy, "The version with copy is not implemented yet..."


    sparse_struct.x = csc_mat_rval
    sparse_struct.z = csc_mat_ival

    sparse_struct.xtype = CHOLMOD_ZOMPLEX                    # CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    sparse_struct.dtype = CHOLMOD_DOUBLE
    # TODO: test if indices are really sorted or not
    sparse_struct.sorted = 0                                 # TRUE if columns are sorted, FALSE otherwise
    sparse_struct.packed = 1                                 # We use the packed CSC version



########################################################################################################################
# CHOLMOD
########################################################################################################################

def cholmod_version():
    version_string = "CHOLMOD version %s" % CHOLMOD_VERSION

    return version_string

def cholmod_detailed_version():
    version_string = "%s.%s.%s (%s)" % (CHOLMOD_MAIN_VERSION,
                                         CHOLMOD_SUB_VERSION,
                                         CHOLMOD_SUBSUB_VERSION,
                                         CHOLMOD_DATE)
    return version_string

cdef class CholmodContext_INT64_t_COMPLEX128_t:
    """
    Cholmod Context from SuiteSparse.

    This version **only** deals with ``LLSparseMatrix_INT64_t_COMPLEX128_t`` objects.

    We follow the common use of Cholmod. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Cholmod.
    """
    CHOLMOD_VERSION = "%s.%s.%s (%s)" % (CHOLMOD_MAIN_VERSION,
                                     CHOLMOD_SUB_VERSION,
                                     CHOLMOD_SUBSUB_VERSION,
                                     CHOLMOD_DATE)

    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, LLSparseMatrix_INT64_t_COMPLEX128_t A):
        """
        """
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz

        # test if we can use CHOLMOD
        assert self.nrow == self.ncol, "Only square matrices are handled in CHOLMOD"
        # TODO: change this. This is an assumption that is too strong
        assert self.A.is_symmetric, "Only symmetric matrices (using the symmetric storage scheme) are handled in CHOLMOD"

        self.csc_mat = self.A.to_csc()

        # CHOLMOD
        self.common_struct = cholmod_common()
        cholmod_l_start(&self.common_struct)

        # TODO: add the possibility to use directly a CSCSparseMatrix
        self.sparse_struct = cholmod_sparse()
        # common attributes for real and complex matrices
        populate1_cholmod_sparse_struct_with_CSCSparseMatrix(&self.sparse_struct, self.csc_mat)


        cdef:
            FLOAT64_t * rval
            FLOAT64_t * ival

        rval = <FLOAT64_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT64_t))
        if not rval:
            raise MemoryError()
        self.csc_rval = rval

        ival = <FLOAT64_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT64_t))
        if not ival:
            PyMem_Free(rval)
            raise MemoryError()
        self.csc_ival = ival

        # split array of complex values into two real value arrays
        split_array_complex_values_kernel_INT64_t_COMPLEX128_t(self.csc_mat.val, self.nnz,
                                                     self.csc_rval, self.nnz,
                                                     self.csc_ival, self.nnz)


        # TODO: uncomment when everything is OK
        #cholmod_l_check_sparse(&self.sparse_struct, &self.common_struct)

    ####################################################################################################################
    # Properties
    ####################################################################################################################
    # Propreties that bear the same name as a reserved Python keyword, are prefixed by 'c_'.
    ######################################### COMMON STRUCT Properties #################################################
    # Printing
    property c_print:
        def __get__(self): return self.common_struct.print_
        def __set__(self, value): self.common_struct.print_ = value

    property precise:
        def __get__(self): return self.common_struct.precise
        def __set__(self, value): self.common_struct.precise = value

    property try_catch:
        def __get__(self): return self.common_struct.try_catch
        def __set__(self, value): self.common_struct.try_catch = value

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """

        """
        cholmod_l_finish(&self.common_struct)
        Py_DECREF(self.A) # release ref

    ####################################################################################################################
    # COMMON OPERATIONS
    ####################################################################################################################
    def reset_default_parameters(self):
        cholmod_l_defaults(&self.common_struct)

    ####################################################################################################################
    # GPU
    ####################################################################################################################
    def request_GPU(self):
        """
        GPU-acceleration is requested.

        If GPU processing is requested but there is no GPU present, CHOLMOD will continue using the CPU only.
        Consequently it is **always safe** to request GPU processing.

        """
        self.common_struct.useGPU = 1

    def prohibit_GPU(self):
        """
        GPU-acceleration is explicitely prohibited.

        """
        self.common_struct.useGPU = 0

    ####################################################################################################################
    # PRINTING
    ####################################################################################################################
    def print_common_struct(self):
        cholmod_l_print_common("cholmod_common_struct", &self.common_struct)

    def print_sparse_matrix(self):
        cholmod_l_print_sparse(&self.sparse_struct, "cholmod_sparse_matrix", &self.common_struct)