from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT64_t cimport LLSparseMatrix_INT64_t_FLOAT64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT64_t cimport CSCSparseMatrix_INT64_t_FLOAT64_t

import numpy as np
cimport numpy as cnp

from  cysparse.linalg.suitesparse.cholmod.cholmod_INT64_t_FLOAT64_t cimport *


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

    ####################################################################################################################
    # EXPERT MODE
    ####################################################################################################################
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

cdef class SPQRContext_INT64_t_FLOAT64_t:
    """
    SPQR Context from SuiteSparse.

    This version **only** deals with ``LLSparseMatrix_INT64_t_FLOAT64_t`` objects.

    We follow the common use of SPQR. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in SPQR.
    """
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, LLSparseMatrix_INT64_t_FLOAT64_t A, bint verbose=False):
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
        cholmod_l_start(&self.common_struct)

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


        cholmod_l_finish(&self.common_struct)

        Py_DECREF(self.A) # release ref

    ####################################################################################################################
    # COMMON OPERATIONS
    ####################################################################################################################
    cdef bint _create_symbolic(self, int ordering, bint allow_tol):
        """
        Create the symbolic object.

        Note:
            Create the object no matter what. See :meth:`create_symbolic` for a conditional creation.

        """
        cdef bint status = 1
        self.factors_struct_initialized = True

        self.factors_struct = SuiteSparseQR_C_symbolic(ordering, allow_tol, &self.sparse_struct, &self.common_struct)

        if self.factors_struct is NULL:
            status = 0
            self.factors_struct_initialized = False

        return status


    def create_symbolic(self, ordering=SPQR_ORDERING_BEST, rank_detection=False):
        """
        Create the symbolic object if it is not already in cache.

        Args:
            ordering:
            rank_detection:

        Note:
            We don't allow to force recomputation of the ``factors_struct`` because we can not delete ``factors_struct``
            without deleting at the same time ``common_struct``.

        """
        if self.factors_struct_initialized:
            return

        cdef bint status = self._create_symbolic(ordering, rank_detection)

        # TODO: raise exception

    cdef bint _create_numeric(self, double drop_tol):
        """
        Create the numeric object.

        Args:
            drop_tol: Treat columns with 2-norm <= drop_tol as zero.

        Note:
            Create the object no matter what. See :meth:`create_numeric` for a conditional creation.
            No test is done to verify if ``create_symbolic()`` has been called before.

        """
        cdef int status = 0

        SuiteSparseQR_C_numeric(drop_tol, &self.sparse_struct, self.factors_struct, &self.common_struct)
        self.numeric_computed = True

        return status

    def create_numeric(self, drop_tol=0):
        """
        Create the numeric object if it is not already in cache.

        Args:

        """
        if self.numeric_computed:
            return

        self.create_symbolic()

        cdef int status = self._create_numeric(drop_tol)

        # TODO: raise exception on error


    def analyze(self, ordering=SPQR_ORDERING_BEST, rank_detection=False, drop_tol=0):
        self.create_symbolic(ordering, rank_detection)
        self.create_numeric(drop_tol)

        # TODO: raise exception...


    def factorize(self, force = False, ordering=SPQR_ORDERING_BEST, drop_tol=0):
        self.factorized = True

        # if needed
        self.analyze(ordering=ordering)

        if not self.factorized or force:
            self.factors_struct = SuiteSparseQR_C_factorize(ordering, drop_tol, &self.sparse_struct, &self.common_struct)

        if self.factors_struct is NULL:
            status = 0
            self.factors_struct_initialized = False
            self.factorized = False


        # TODO: raise exception if needed...
        #return status

    cdef _SPQR_istat(self):
        s = ''

        # ordering
        s += 'ORDERING USED: %s' % ORDERING_METHOD_LIST[self.common_struct.SPQR_istat[7]]

        return s

    def spqr_statistics(self):
        # TODO: todo
        return self. _SPQR_istat()




















































