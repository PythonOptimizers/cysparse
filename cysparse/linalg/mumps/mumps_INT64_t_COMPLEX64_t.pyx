from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX64_t cimport LLSparseMatrix_INT64_t_COMPLEX64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX64_t cimport CSCSparseMatrix_INT64_t_COMPLEX64_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

import numpy as np
cimport numpy as cnp

cnp.import_array()




cdef class MumpsContext_INT64_t_COMPLEX64_t:
    """
    Mumps Context.

    This version **only** deals with ``LLSparseMatrix_INT64_t_COMPLEX64_t`` objects.

    We follow the common use of Mumps. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Mumps.
    """
    MUMPS_VERSION = "0.66666"

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
