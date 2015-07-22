"""
Factory method to access Mumps.

The python code (this module) is autmotically generated because the code depends
on the compile/architecture configuration.

"""

from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
from cysparse.types.cysparse_types import *


    
from cysparse.linalg.mumps.mumps_INT32_t_FLOAT32_t import MumpsContext_INT32_t_FLOAT32_t
    
from cysparse.linalg.mumps.mumps_INT32_t_FLOAT64_t import MumpsContext_INT32_t_FLOAT64_t
    


def NewMumpsContext(A, verbose=False):
    """
    Create and return the right Mumps context object.

    Args:
        A: :class:`LLSparseMatrix`.
    """
    if not PyLLSparseMatrix_Check(A):
        raise TypeError('Matrix A should be a LLSparseMatrix')

    itype = A.itype
    dtype = A.dtype


    
    if itype == INT32_T:
    
        
        if dtype == FLOAT32_T:
        
            return MumpsContext_INT32_t_FLOAT32_t(A, verbose=verbose)
    
        
        elif dtype == FLOAT64_T:
        
            return MumpsContext_INT32_t_FLOAT64_t(A, verbose=verbose)
    
    


    allowed_types = '\titype:INT32_T\n\tdtype:FLOAT32_T,FLOAT64_T\n'

    type_error_msg = 'Matrix has an index and/or element type that is incompatible with Mumps\nAllowed types:\n%s' % allowed_types
    raise TypeError(type_error_msg)