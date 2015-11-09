from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc,free, calloc
from cpython cimport Py_INCREF, Py_DECREF

from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX128_t cimport LLSparseMatrix_INT64_t_COMPLEX128_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX128_t cimport CSCSparseMatrix_INT64_t_COMPLEX128_t

import numpy as np
cimport numpy as cnp

from collections import OrderedDict

from  cysparse.linalg.suitesparse.cholmod.cholmod_INT64_t_COMPLEX128_t cimport *

from cysparse.types.cysparse_generic_types cimport split_array_complex_values_kernel_INT64_t_COMPLEX128_t, join_array_complex_values_kernel_INT64_t_COMPLEX128_t


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

ORDERING_METHOD_DICT = OrderedDict()
ORDERING_METHOD_DICT['SPQR_ORDERING_FIXED'] = SPQR_ORDERING_FIXED
ORDERING_METHOD_DICT['SPQR_ORDERING_NATURAL'] = SPQR_ORDERING_NATURAL
ORDERING_METHOD_DICT['SPQR_ORDERING_COLAMD'] = SPQR_ORDERING_COLAMD
ORDERING_METHOD_DICT['SPQR_ORDERING_GIVEN'] = SPQR_ORDERING_GIVEN
ORDERING_METHOD_DICT['SPQR_ORDERING_CHOLMOD'] = SPQR_ORDERING_CHOLMOD
ORDERING_METHOD_DICT['SPQR_ORDERING_AMD'] = SPQR_ORDERING_AMD
ORDERING_METHOD_DICT['SPQR_ORDERING_METIS'] = SPQR_ORDERING_METIS
ORDERING_METHOD_DICT['SPQR_ORDERING_DEFAULT'] = SPQR_ORDERING_DEFAULT
ORDERING_METHOD_DICT['SPQR_ORDERING_BEST'] = SPQR_ORDERING_BEST
ORDERING_METHOD_DICT['SPQR_ORDERING_BESTAMD'] = SPQR_ORDERING_BESTAMD

ORDERING_METHOD_LIST = ORDERING_METHOD_DICT.keys()




SPQR_SYS_DICT = {
        'SPQR_RX_EQUALS_B'     : SPQR_RX_EQUALS_B,
        'SPQR_RETX_EQUALS_B'   : SPQR_RETX_EQUALS_B,
        'SPQR_RTX_EQUALS_B'    : SPQR_RTX_EQUALS_B,
        'SPQR_RTX_EQUALS_ETB'  : SPQR_RTX_EQUALS_ETB
    }



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



cdef class SPQRContext_INT64_t_COMPLEX128_t:
    """
    SPQR Context from SuiteSparse.

    This version **only** deals with ``LLSparseMatrix_INT64_t_COMPLEX128_t`` objects.

    We follow the common use of SPQR. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in SPQR.
    """
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, LLSparseMatrix_INT64_t_COMPLEX128_t A, bint verbose=False):
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
    def solve(self, cnp.ndarray[cnp.npy_complex128, mode="c"] b, str ordering='SPQR_ORDERING_BEST', double drop_tol=0):
        """
        Solve `A*x = b` with `b` dense (case `X = A\B`).

        Args:
            ordering (str):
            drop_tol (double): Treat columns with 2-norm <= drop_tol as zero.

        """
        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")

        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        # TODO: change this (see TODO below)
        if b.ndim != 1:
            raise NotImplementedError('Matrices for right member will be implemented soon...')

        # convert NumPy array to CHOLMOD dense vector
        cdef cholmod_dense B

        # TODO: does it use multidimension (matrix and not vector)
        # Currently ONLY support vectors...
        B = numpy_ndarray_to_cholmod_dense(b)

        cdef cholmod_dense * cholmod_sol

        cholmod_sol = SuiteSparseQR_C_backslash(ORDERING_METHOD_DICT[ordering], drop_tol, &self.sparse_struct, &B, &self.common_struct)

        # test solution
        if cholmod_sol == NULL:
            # no solution was found
            raise RuntimeError('No solution found')

        # TODO: free B
        # TODO: convert sol to NumPy array

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        # make a copy
        cdef INT64_t j

        cdef COMPLEX128_t * cholmod_sol_array_ptr = <COMPLEX128_t * > cholmod_sol.x


        raise NotImplementedError("To be coded")


        # Free CHOLMOD dense solution
        cholmod_l_free_dense(&cholmod_sol, &self.common_struct)

        return sol

    def solve_default(self, cnp.ndarray[cnp.npy_complex128, mode="c"] b):
        """
        Solve `A*x = b` with `b` dense (case `X = A\B`).

        Args:
            b

        """
        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")

        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        # TODO: change this (see TODO below)
        if b.ndim != 1:
            raise NotImplementedError('Matrices for right member will be implemented soon...')

        # convert NumPy array to CHOLMOD dense vector
        cdef cholmod_dense B

        # TODO: does it use multidimension (matrix and not vector)
        # Currently ONLY support vectors...
        B = numpy_ndarray_to_cholmod_dense(b)

        cdef cholmod_dense * cholmod_sol

        cholmod_sol = SuiteSparseQR_C_backslash_default(&self.sparse_struct, &B, &self.common_struct)

        # test solution
        if cholmod_sol == NULL:
            # no solution was found
            raise RuntimeError('No solution found')

        # TODO: free B
        # TODO: convert sol to NumPy array

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        # make a copy
        cdef INT64_t j

        cdef COMPLEX128_t * cholmod_sol_array_ptr = <COMPLEX128_t * > cholmod_sol.x


        raise NotImplementedError("To be coded")


        # Free CHOLMOD dense solution
        cholmod_l_free_dense(&cholmod_sol, &self.common_struct)

        return sol

    def get_QR(self, SuiteSparse_long econ, str ordering='SPQR_ORDERING_BEST', double drop_tol = 0):
        """
        Return QR factorisation objects. If needed, the QR factorisation is triggered.

        Args:
            ordering (str): SPQR ordening. See ``ORDERING_METHOD_LIST``.
            drop_tol (double): Columns with ``2-norm <= drop_tol`` are treated as 0.
            econ (SuiteSparse_long): Parameter such that ``e = max(min(m,econ),rank(A))``.

        Returns:
            ``(Q,R,E)``

            The original matrix ``A`` is factorized into

                ``Q R = A E``

            where:
             - ``A`` is a ``m``-by-``n`` sparse matrix to factorize;
             - ``Q`` is a ``m``-by-``e`` sparse matrix;
             - ``R`` is a ``e``-by-``n`` sparse matrix and
             - ``E`` is a ``n``-by-``n`` permutation matrice.

            ``Q`` and ``R`` are returned as ``CSCSparseMatrix`` matrices.
            ``E`` is returned as a one dimensional NumPy vector of size ``n``.

        Warning:
            We don't cache any matrix. ``E`` is **always** returned.

        """
        cdef:
            cholmod_sparse *Q_cholmod
            cholmod_sparse *R_cholmod
            SuiteSparse_long *E_cholmod
            SuiteSparse_long status

        # returns rank(A) est., (-1) if failure
        status = SuiteSparseQR_C_QR (ORDERING_METHOD_DICT[ordering], drop_tol, econ, &self.sparse_struct, &Q_cholmod, &R_cholmod, &E_cholmod, &self.common_struct)

        if status == -1:
            raise RuntimeError('SPQR could not factorize matrix...')

        # create matrices to return
        cdef:
            CSCSparseMatrix_INT64_t_COMPLEX128_t Q_csc_mat
            CSCSparseMatrix_INT64_t_COMPLEX128_t R_csc_mat

        Q_csc_mat = cholmod_sparse_to_CSCSparseMatrix_INT64_t_COMPLEX128_t(Q_cholmod, no_copy=False)
        R_csc_mat = cholmod_sparse_to_CSCSparseMatrix_INT64_t_COMPLEX128_t(R_cholmod, no_copy=False)

        # create permutation matrix E
        cdef:
            cnp.ndarray[cnp.npy_int64, ndim=1] E_ndarray
            INT64_t i

        # test if E is identity
        if E_cholmod == NULL:
            E_ndarray = np.empty(self.ncol, dtype=np.int64)
            for i from 0 <= i < self.ncol:
                E_ndarray[i] = i
        else:
            E_ndarray = np.empty(self.ncol, dtype=np.int64)

            for i from 0 <= i < self.ncol:
                E_ndarray[i] = E_cholmod[i]

        # delete cholmod matrices
        cholmod_l_free_sparse(&Q_cholmod, &self.common_struct)
        cholmod_l_free_sparse(&R_cholmod, &self.common_struct)

        cholmod_l_free(self.ncol, sizeof(SuiteSparse_long),	E_cholmod,  &self.common_struct)

        return Q_csc_mat, R_csc_mat, E_ndarray

    ####################################################################################################################
    # EXPERT MODE
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


    def create_symbolic(self, str ordering='SPQR_ORDERING_BEST', rank_detection=False):
        """
        Create the symbolic object if it is not already in cache.

        Args:
            ordering (str):
            rank_detection:

        Note:
            We don't allow to force recomputation of the ``factors_struct`` because we can not delete ``factors_struct``
            without deleting at the same time ``common_struct``.

        """
        if self.factors_struct_initialized:
            return

        cdef bint status = self._create_symbolic(ORDERING_METHOD_DICT[ordering], rank_detection)

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


    def analyze(self, str ordering='SPQR_ORDERING_BEST', rank_detection=False, drop_tol=0):
        self.create_symbolic(ordering, rank_detection)
        self.create_numeric(drop_tol)

        # TODO: raise exception...


    def factorize(self, force = False, str ordering='SPQR_ORDERING_BEST', drop_tol=0):
        self.factorized = True

        # if needed
        self.analyze(ordering)

        if not self.factorized or force:
            self.factors_struct = SuiteSparseQR_C_factorize(ORDERING_METHOD_DICT[ordering], drop_tol, &self.sparse_struct, &self.common_struct)

        if self.factors_struct is NULL:
            status = 0
            self.factors_struct_initialized = False
            self.factorized = False


        # TODO: raise exception if needed...
        #return status

    def solve_expert(self, cnp.ndarray[cnp.npy_complex128, mode="c"] b, spqr_sys='SPQR_RX_EQUALS_B'):
        """
        This is the expert solve (simply called `solve()` in :program:`SPQR`).

        Args:
            spqr_sys: can be:
                - SPQR_RX_EQUALS_B =  0       # solve R*X=B      or X = R\B
                - SPQR_RETX_EQUALS_B = 1      # solve R*E'*X=B   or X = E*(R\B)
                - SPQR_RTX_EQUALS_B = 2       # solve R'*X=B     or X = R'\B
                - SPQR_RTX_EQUALS_ETB = 3     # solve R'*X=E'*B  or X = R'\(E'*B)
        """

        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")

        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        # TODO: change this (see TODO below)
        if b.ndim != 1:
            raise NotImplementedError('Matrices for right member will be implemented soon...')

        if spqr_sys not in SPQR_SYS_DICT.keys():
            raise ValueError('spqr_sys must be in' % SPQR_SYS_DICT.keys())

        # if needed
        self.factorize()


        # convert NumPy array to CHOLMOD dense vector
        cdef cholmod_dense B

        # TODO: does it use multidimension (matrix and not vector)
        # Currently ONLY support vectors...
        B = numpy_ndarray_to_cholmod_dense(b)

        cdef cholmod_dense * cholmod_sol
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        cholmod_sol = SuiteSparseQR_C_solve(SPQR_SYS_DICT[spqr_sys], self.factors_struct, &B, &self.common_struct)

        #cholmod_sol = cholmod_l_solve(CHOLMOD_SYS_DICT[cholmod_sys], self.factor_struct, &B, &self.common_struct)

        # TODO: free B
        # TODO: convert sol to NumPy array

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        # make a copy
        cdef INT64_t j

        cdef COMPLEX128_t * cholmod_sol_array_ptr = <COMPLEX128_t * > cholmod_sol.x


        raise NotImplementedError("To be coded")


        # Free CHOLMOD dense solution
        cholmod_l_free_dense(&cholmod_sol, &self.common_struct)

        return sol


    ####################################################################################################################
    # PARAMETERS
    ####################################################################################################################
    def SPQR_ordering_list(self):
        return ORDERING_METHOD_LIST

    ####################################################################################################################
    # STATISTICS
    ####################################################################################################################
    def SPQR_orderning(self):
        """
        Returns the chosen ordering.
        """
        return ORDERING_METHOD_LIST[self.common_struct.SPQR_istat[7]]

    def SPQR_drop_tol_used(self):
        """
        Return `drop_tol` (`double`). columns with 2-norm <= tol treated as 0
        """
        return self.common_struct.SPQR_tol_used

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