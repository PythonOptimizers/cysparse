from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

# external definition of this type
ctypedef long SuiteSparse_long # This is exactly CySparse's INT64_t

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

    ctypedef struct cholmod_common:
        # parameters for symbolic/numeric factorization and update/downdate
        double dbound
        double grow0
        double grow1
        size_t grow2
        size_t maxrank
        double supernodal_switch
        int supernodal
        int final_asis
        int final_super
        int final_ll
        int final_pack
        int final_monotonic
        int final_resymbol
        double zrelax [3]
        size_t nrelax [3]
        int prefer_zomplex
        int prefer_upper
        int quick_return_if_not_posdef
        int prefer_binary

        # printing and error handling options

        # workspace
        size_t nrow
        SuiteSparse_long mark

    int cholmod_start(cholmod_common *Common)
    int cholmod_finish(cholmod_common *Common)

def cholmod_version():
    version_string = "CHOLMOD version %s" % CHOLMOD_VERSION

    return version_string

def cholmod_detailed_version():
    version_string = "%s.%s.%s (%s)" % (CHOLMOD_MAIN_VERSION,
                                         CHOLMOD_SUB_VERSION,
                                         CHOLMOD_SUBSUB_VERSION,
                                         CHOLMOD_DATE)
    return version_string

cdef class CholmodContext_INT32_t_COMPLEX128_t:
    """
    Cholmod Context from SuiteSparse.

    This version **only** deals with ``LLSparseMatrix_INT32_t_COMPLEX128_t`` objects.

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
    def __cinit__(self, LLSparseMatrix_INT32_t_COMPLEX128_t A):
        """
        """
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz

        # test if we can use CHOLMOD
        assert self.nrow == self.ncol, "Only square matrices are handled in CHOLMOD"
        assert self.A.is_symmetric, "Only symmetric matrices (using the symmetric storage scheme) are handled in CHOLMOD"


        cholmod_start(&self.common_struct)


    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """

        """
        cholmod_finish(&self.common_struct)
        Py_DECREF(self.A) # release ref