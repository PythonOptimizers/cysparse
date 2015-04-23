from sparse_lib.sparse.ll_mat cimport LLSparseMatrix
from sparse_lib.sparse.csc_mat cimport CSCSparseMatrix

cdef extern from "umfpack.h":
    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO



cdef class UmfpackSolver:
    cdef:
        LLSparseMatrix A

        int nrow
        int ncol

        # Matrix A in CSC format
        #double * val
        #int * col
        #int * ind
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

