from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat cimport LLSparseMatrix
from cysparse.sparse.csc_mat cimport CSCSparseMatrix


cdef extern from "umfpack.h":
    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO


cdef class UmfpackSolver:
    cdef:
        LLSparseMatrix A

        SIZE_t nrow
        SIZE_t ncol

        public bint is_complex

        public str family

        # Matrix A in CSC format
        CSCSparseMatrix csc_mat

        # UMFPACK opaque objects
        void * symbolic
        bint symbolic_computed

        void * numeric
        bint numeric_computed

        # Control and Info arrays
        public double info[UMFPACK_INFO]
        public double control[UMFPACK_CONTROL]

    cdef int _create_symbolic(self)
    cdef int _create_numeric(self)

