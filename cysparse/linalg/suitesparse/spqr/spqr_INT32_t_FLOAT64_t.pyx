from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t cimport LLSparseMatrix_INT32_t_FLOAT64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT64_t cimport CSCSparseMatrix_INT32_t_FLOAT64_t

import numpy as np
cimport numpy as cnp

from  cysparse.linalg.suitesparse.cholmod.cholmod_INT32_t_FLOAT64_t cimport *


cdef extern from "SuiteSparseQR_definitions.h":
    # ordering options
    cdef enum:
        SPQR_ORDERING_FIXED = 0
        SPQR_ORDERING_NATURAL = 1
        SPQR_ORDERING_COLAMD = 2
        SPQR_ORDERING_GIVEN = 3       # only used for C/C++ interface
        SPQR_ORDERING_CHOLMOD = 4     # CHOLMOD best-effort (COLAMD, METIS,...)
        SPQR_ORDERING_AMD = 5         # AMD(A'*A)
        SPQR_ORDERING_METIS = 6       # metis(A'*A)
        SPQR_ORDERING_DEFAULT = 7     # SuiteSparseQR default ordering
        SPQR_ORDERING_BEST = 8        # try COLAMD, AMD, and METIS; pick best
        SPQR_ORDERING_BESTAMD = 9     # try COLAMD and AMD; pick best

    # tol options
    cdef enum:
        SPQR_DEFAULT_TOL = -2       # if tol <= -2, the default tol is used
        SPQR_NO_TOL = -1            # if -2 < tol < 0, then no tol is used

    # for qmult, method can be 0,1,2,3:
    cdef enum:
        SPQR_QTX = 0
        SPQR_QX  = 1
        SPQR_XQT = 2
        SPQR_XQ  = 3

    # system can be 0,1,2,3:  Given Q*R=A*E from SuiteSparseQR_factorize:
    cdef enum:
        SPQR_RX_EQUALS_B =  0       # solve R*X=B      or X = R\B
        SPQR_RETX_EQUALS_B = 1      # solve R*E'*X=B   or X = E*(R\B)
        SPQR_RTX_EQUALS_B = 2       # solve R'*X=B     or X = R'\B
        SPQR_RTX_EQUALS_ETB = 3     # solve R'*X=E'*B  or X = R'\(E'*B)

ORDERING_METHOD_LIST = ['SPQR_ORDERING_FIXED',
        'SPQR_ORDERING_NATURAL',
        'SPQR_ORDERING_COLAMD',
        'SPQR_ORDERING_GIVEN',
        'SPQR_ORDERING_CHOLMOD',
        'SPQR_ORDERING_AMD',
        'SPQR_ORDERING_METIS',
        'SPQR_ORDERING_DEFAULT',
        'SPQR_ORDERING_BEST',
        'SPQR_ORDERING_BESTAMD']



cdef extern from  "SuiteSparseQR_C.h":
    # returns rank(A) estimate, (-1) if failure
    cdef SuiteSparse_long SuiteSparseQR_C(
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        SuiteSparse_long econ,      # e = max(min(m,econ),rank(A))
        int getCTX,                 # 0: Z=C (e-by-k), 1: Z=C', 2: Z=X (e-by-k)
        cholmod_sparse *A,          # m-by-n sparse matrix to factorize
        cholmod_sparse *Bsparse,    # sparse m-by-k B
        cholmod_dense  *Bdense,     # dense  m-by-k B
        # outputs:
        cholmod_sparse **Zsparse,   # sparse Z
        cholmod_dense  **Zdense,    # dense Z
        cholmod_sparse **R,         # e-by-n sparse matrix
        SuiteSparse_long **E,       # size n column perm, NULL if identity
        cholmod_sparse **H,         # m-by-nh Householder vectors
        SuiteSparse_long **HPinv,   # size m row permutation
        cholmod_dense **HTau,       # 1-by-nh Householder coefficients
        cholmod_common *cc          # workspace and parameters
        )


    # [Q,R,E] = qr(A), returning Q as a sparse matrix
    # returns rank(A) est., (-1) if failure
    cdef SuiteSparse_long SuiteSparseQR_C_QR (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        SuiteSparse_long econ,      # e = max(min(m,econ),rank(A))
        cholmod_sparse *A,          # m-by-n sparse matrix to factorize
        # outputs:
        cholmod_sparse **Q,         # m-by-e sparse matrix
        cholmod_sparse **R,         # e-by-n sparse matrix
        SuiteSparse_long **E,       # size n column perm, NULL if identity
        cholmod_common *cc          # workspace and parameters
        )

    # X = A\B where B is dense
    # returns X, NULL if failure
    cdef cholmod_dense *SuiteSparseQR_C_backslash (
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_dense  *B,          # m-by-k
        cholmod_common *cc          # workspace and parameters
    )

    # X = A\B where B is dense, using default ordering and tol
    # returns X, NULL if failure
    cdef cholmod_dense *SuiteSparseQR_C_backslash_default (
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_dense  *B,          # m-by-k
        cholmod_common *cc          # workspace and parameters
    )

    # X = A\B where B is sparse
    # returns X, or NULL
    cdef cholmod_sparse *SuiteSparseQR_C_backslash_sparse (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_sparse *B,          # m-by-k
        cholmod_common *cc          # workspace and parameters
    )



cdef class SPQRContext_INT32_t_FLOAT64_t:
    """
    SPQR Context from SuiteSparse.

    This version **only** deals with ``LLSparseMatrix_INT32_t_FLOAT64_t`` objects.

    We follow the common use of SPQR. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in SPQR.
    """
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, LLSparseMatrix_INT32_t_FLOAT64_t A, bint verbose=False):
        """
        """
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz

        self.csc_mat = self.A.to_csc()

        # CHOLMOD
        self.common_struct = cholmod_common()
        cholmod_start(&self.common_struct)

        self.sparse_struct = cholmod_sparse()
        # common attributes for real and complex matrices
        populate1_cholmod_sparse_struct_with_CSCSparseMatrix(&self.sparse_struct, self.csc_mat)

        populate2_cholmod_sparse_struct_with_CSCSparseMatrix(&self.sparse_struct, self.csc_mat)


        self.factors_struct_initialized = False
        self.numeric_computed = False
        self.factorized = False


    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """

        """
        # we don't delete sparse_struct as **all** arrays are allocated in self.csc_mat
        # TODO: doesn't work... WHY?
        #del self.csc_mat


        cholmod_finish(&self.common_struct)

        Py_DECREF(self.A) # release ref

    ####################################################################################################################
    # COMMON OPERATIONS
    ####################################################################################################################


    ####################################################################################################################
    # STATISTICS
    ####################################################################################################################
    def SPQR_orderning(self):
        """
        Returns the chosen ordering.
        """
        return ORDERING_METHOD_LIST[self.common_struct.SPQR_istat[7]]

    cdef _SPQR_istat(self):
        """
        Main statistic method for SPQR, :program:`Cython` version.
        """
        s = ''

        # ordering
        s += 'ORDERING USED: %s' % self.SPQR_orderning()

        return s

    def spqr_statistics(self):
        """
        Main statistic for SPQR.

        """
        # TODO: todo
        return self. _SPQR_istat()