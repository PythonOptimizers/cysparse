"""
Factory method to access SPQR.

The python code (this module) is autmotically generated because the code depends
on the compile/architecture configuration.

"""

from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
from cysparse.cysparse_types.cysparse_types import *


    
from cysparse.linalg.suitesparse.spqr.spqr_INT32_t_FLOAT64_t import SPQRContext_INT32_t_FLOAT64_t
    
from cysparse.linalg.suitesparse.spqr.spqr_INT32_t_COMPLEX128_t import SPQRContext_INT32_t_COMPLEX128_t
    

    
from cysparse.linalg.suitesparse.spqr.spqr_INT64_t_FLOAT64_t import SPQRContext_INT64_t_FLOAT64_t
    
from cysparse.linalg.suitesparse.spqr.spqr_INT64_t_COMPLEX128_t import SPQRContext_INT64_t_COMPLEX128_t
    


def NewSPQRContext(A, verbose=False):
    """
    Create and return the right SPQR context object.

    Args:
        A: :class:`LLSparseMatrix`.
    """
    if not PyLLSparseMatrix_Check(A):
        raise TypeError('Matrix A should be a LLSparseMatrix')

    itype = A.itype
    dtype = A.dtype


    
    if itype == INT32_T:
    
        
        if dtype == FLOAT64_T:
        
            return SPQRContext_INT32_t_FLOAT64_t(A, verbose=verbose)
    
        
        elif dtype == COMPLEX128_T:
        
            return SPQRContext_INT32_t_COMPLEX128_t(A, verbose=verbose)
    
    

    
    elif itype == INT64_T:
    
        
        if dtype == FLOAT64_T:
        
            return SPQRContext_INT64_t_FLOAT64_t(A, verbose=verbose)
    
        
        elif dtype == COMPLEX128_T:
        
            return SPQRContext_INT64_t_COMPLEX128_t(A, verbose=verbose)
    
    


    allowed_types = '\titype:INT32_T,INT64_T\n\tdtype:FLOAT64_T,COMPLEX128_T\n'

    type_error_msg = 'Matrix has an index and/or element type that is incompatible with SPQR\nAllowed types:\n%s' % allowed_types
    raise TypeError(type_error_msg)