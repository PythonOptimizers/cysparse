from cysparse.cysparse_types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX128_t cimport LLSparseMatrix_INT64_t_COMPLEX128_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX128_t cimport CSCSparseMatrix_INT64_t_COMPLEX128_t

import numpy as np
cimport numpy as cnp

# external definition of this type
ctypedef long SuiteSparse_long # This is exactly CySparse's INT64_t


from  cysparse.linalg.suitesparse.cholmod.cholmod_INT64_t_COMPLEX128_t cimport *

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


    ####################################################################################################################
    # EXPERT MODE
    ####################################################################################################################
    # A real or complex QR factorization, computed by SuiteSparseQR_C_factorize
    ctypedef struct SuiteSparseQR_C_factorization:
        int xtype                  # CHOLMOD_REAL or CHOLMOD_COMPLEX
        void *factors              # from SuiteSparseQR_factorize <double> or SuiteSparseQR_factorize <Complex>


    cdef SuiteSparseQR_C_factorization *SuiteSparseQR_C_factorize (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        double tol,                 # columns with 2-norm <= tol treated as 0
        cholmod_sparse *A,          # m-by-n sparse matrix
        cholmod_common *cc          # workspace and parameters
    ) 

    cdef SuiteSparseQR_C_factorization *SuiteSparseQR_C_symbolic (
        # inputs:
        int ordering,               # all, except 3:given treated as 0:fixed
        int allow_tol,              # if TRUE allow tol for rank detection
        cholmod_sparse *A,          # m-by-n sparse matrix, A->x ignored
        cholmod_common *cc          # workspace and parameters
    ) 

    cdef int SuiteSparseQR_C_numeric (
        # inputs:
        double tol,                 # treat columns with 2-norm <= tol as zero
        cholmod_sparse *A,          # sparse matrix to factorize
        # input/output:
        SuiteSparseQR_C_factorization *QR,
        cholmod_common *cc          # workspace and parameters
    ) 

    # Free the QR factors computed by SuiteSparseQR_C_factorize
    # returns TRUE (1) if OK, FALSE (0) otherwise
    cdef int SuiteSparseQR_C_free (
        SuiteSparseQR_C_factorization **QR,
        cholmod_common *cc          # workspace and parameters
    ) 

    # returnx X, or NULL if failure
    cdef cholmod_dense* SuiteSparseQR_C_solve (
        int system,                 # which system to solve
        SuiteSparseQR_C_factorization *QR,  # of an m-by-n sparse matrix A
        cholmod_dense *B,           # right-hand-side, m-by-k or n-by-k
        cholmod_common *cc          # workspace and parameters
    ) 

    # Applies Q in Householder form (as stored in the QR factorization object
    # returned by SuiteSparseQR_C_factorize) to a dense matrix X.
    #
    # method SPQR_QTX (0): Y = Q'*X
    # method SPQR_QX  (1): Y = Q*X
    # method SPQR_XQT (2): Y = X*Q'
    # method SPQR_XQ  (3): Y = X*Q

    # returns Y, or NULL on failure
    cdef cholmod_dense *SuiteSparseQR_C_qmult (
        # inputs:
        int method,                 # 0,1,2,3
        SuiteSparseQR_C_factorization *QR,  # of an m-by-n sparse matrix A
        cholmod_dense *X,           # size m-by-n with leading dimension ldx
        cholmod_common *cc          # workspace and parameters
    ) 



cdef class SPQRContext_INT64_t_COMPLEX128_t:
    cdef:
        LLSparseMatrix_INT64_t_COMPLEX128_t A

        INT64_t nrow
        INT64_t ncol
        INT64_t nnz

        # Matrix A in CSC format
        CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat

        cholmod_common common_struct
        cholmod_sparse sparse_struct


        SuiteSparseQR_C_factorization * factors_struct
        bint factors_struct_initialized
        bint numeric_computed
        bint factorized



        # we keep internally two arrays for the complex numbers: this is required by CHOLMOD...
        FLOAT64_t * csc_rval
        FLOAT64_t * csc_ival




    cdef bint _create_symbolic(self, int ordering, bint allow_tol)
    cdef bint _create_numeric(self, double drop_tol)


    cdef _SPQR_istat(self)
