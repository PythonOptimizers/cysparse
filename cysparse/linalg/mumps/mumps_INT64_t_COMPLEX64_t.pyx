
cdef extern from "mumps_c_types.h":

    ctypedef int        MUMPS_INT
    ctypedef np.int8_t  MUMPS_INT8

    ctypedef float      SMUMPS_COMPLEX
    ctypedef float      SMUMPS_REAL

    ctypedef double     DMUMPS_COMPLEX
    ctypedef double     DMUMPS_REAL

    ctypedef struct mumps_complex:
        pass

    ctypedef mumps_complex  CMUMPS_COMPLEX
    ctypedef float          CMUMPS_REAL

    ctypedef struct mumps_double_complex:
        pass

    ctypedef mumps_double_complex  ZMUMPS_COMPLEX
    ctypedef double                ZMUMPS_REAL

    char* MUMPS_VERSION


cdef class MumpsContext_INT64_t_COMPLEX64_t:
    """
    Mumps Context.

    This version **only** deals with ``LLSparseMatrix_INT64_t_COMPLEX64_t`` objects.

    We follow the common use of Mumps. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Mumps.
    """
    MUMPS_VERSION = "%s" % MUMPS_VERSION

    def __cinit__(self, LLSparseMatrix_INT64_t_COMPLEX64_t A):
        """
        Args:
            A: A :class:`LLSparseMatrix_INT64_t_COMPLEX64_t` object.

        Warning:
            The solver takes a "snapshot" of the matrix ``A``, i.e. the results given by the solver are only
            valid for the matrix given. If the matrix ``A`` changes aferwards, the results given by the solver won't
            reflect this change.

        """
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz
