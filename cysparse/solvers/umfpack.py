"""
Factory method to access SuiteSparse UMFPACK.
"""
from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check

from cysparse.solvers.suitesparse.umfpack.umfpack_INT32_t_FLOAT64_t import UmfpackSolver_INT32_t_FLOAT64_t
from cysparse.solvers.suitesparse.umfpack.umfpack_INT32_t_COMPLEX128_t import UmfpackSolver_INT32_t_COMPLEX128_t
from cysparse.solvers.suitesparse.umfpack.umfpack_INT64_t_FLOAT64_t import UmfpackSolver_INT64_t_FLOAT64_t
from cysparse.solvers.suitesparse.umfpack.umfpack_INT64_t_COMPLEX128_t import UmfpackSolver_INT64_t_COMPLEX128_t


def NewUmfpackSolver(A):
    """
    Create and return the right UMFPACK solver object.

    Args:
        A: :class:`LLSparseMatrix`.
    """
    if not PyLLSparseMatrix_Check(A):
        raise TypeError('Matrix A should be a LLSparseMatrix')


    # So few cases that are unlikely to change: we do the dispatch by hand...
    if A.itype == 'INT32_T':
        if A.dtype == 'FLOAT64_T':
            return UmfpackSolver_INT32_t_FLOAT64_t(A)
        elif A.dtype == 'COMPLEX128_T':
            return UmfpackSolver_INT32_t_COMPLEX128_t(A)
        else:
            raise TypeError('Matrix has an element type that is incompatible with Umfpack')

    elif A.itype == 'INT64_T':
        if A.dtype == 'FLOAT64_T':
            return UmfpackSolver_INT64_t_FLOAT64_t(A)
        elif A.dtype == 'COMPLEX128_T':
            return UmfpackSolver_INT64_t_COMPLEX128_t(A)
        else:
            raise TypeError('Matrix has an element type that is incompatible with Umfpack')

    else:
        raise TypeError('Matrix has an index type that is incompatible with Umfpack')