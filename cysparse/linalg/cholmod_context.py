"""
Factory method to access SuiteSparse CHOLMOD.
"""
from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
from cysparse.cysparse_types.cysparse_types import *

from cysparse.linalg.suitesparse.cholmod.cholmod_INT32_t_FLOAT64_t import CholmodContext_INT32_t_FLOAT64_t
from cysparse.linalg.suitesparse.cholmod.cholmod_INT32_t_COMPLEX128_t import CholmodContext_INT32_t_COMPLEX128_t
from cysparse.linalg.suitesparse.cholmod.cholmod_INT64_t_FLOAT64_t import CholmodContext_INT64_t_FLOAT64_t
from cysparse.linalg.suitesparse.cholmod.cholmod_INT64_t_COMPLEX128_t import CholmodContext_INT64_t_COMPLEX128_t

# general functions
from cysparse.linalg.suitesparse.cholmod.cholmod_INT32_t_FLOAT64_t import cholmod_version as cm_version
def cholmod_version():
    return cm_version()

from cysparse.linalg.suitesparse.cholmod.cholmod_INT32_t_FLOAT64_t import cholmod_detailed_version as cm_detailed_version
def cholmod_detailed_version():
    return cm_detailed_version()


def NewCholmodContext(A):
    """
    Create and return the right UMFPACK context object.

    Args:
        A: :class:`LLSparseMatrix`.
    """
    if not PyLLSparseMatrix_Check(A):
        raise TypeError('Matrix A should be a LLSparseMatrix')


    # So few cases that are unlikely to change: we do the dispatch by hand...
    if A.itype == INT32_T:
        if A.dtype == FLOAT64_T:
            return CholmodContext_INT32_t_FLOAT64_t(A)
        elif A.dtype == COMPLEX128_T:
            return CholmodContext_INT32_t_COMPLEX128_t(A)
        else:
            raise TypeError('Matrix has an element type that is incompatible with Cholmod')

    elif A.itype == INT64_T:
        if A.dtype == FLOAT64_T:
            return CholmodContext_INT64_t_FLOAT64_t(A)
        elif A.dtype == COMPLEX128_T:
            return CholmodContext_INT64_t_COMPLEX128_t(A)
        else:
            raise TypeError('Matrix has an element type that is incompatible with Cholmod')

    else:
        raise TypeError('Matrix has an index type that is incompatible with Cholmod')