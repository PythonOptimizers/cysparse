
To create a matrix, we use *factory methods* [#factory_method_strange_name]_: 
functions that return an object corresponding
to their arguments. Different arguments make them return different kind of objects (matrices).

How to create a matrix?
========================

Before you can use any type of sparse matrix, you **must** first instantiate an ``LLSparseMatrix``. This matrix is well suited for construction but is not very optimized for most matrix operations. Once you have an ``LLSparseMatrix``, you can create a specialized sparse matrix from it.

Sparse matrices all come from a ``LLSparseMatrix``
------------------------------------------------------

The ``LLSparseMatrix`` matrix type is the only one that is *mutable*. You can add and/or delete elements, rows, columns, sub-matrices at will. Once you have constructed your matrix, it is time to transform it into an appropriate 
matrix format that is optimized for your needs. This transformation is not done in place and a copy is made. Here is an example:

..  code-block:: python

    A = ...  # A is a LLSparseMatrix
    # add some elements
    for i in range(n):
        for j in range(m):
            A[i, j] = ...
    
    # once the matrix is constructed, transform it into suitable matrix format
    # here to CSC
    C = A.to_csc()

..  _matrices_must_be_instantiated_by_a_factory_method:

``LLSparseMatrix`` matrices must be instantiated by a factory method
---------------------------------------------------------------------------

Matrices **must** be instantiated by one of the factory methods. 
For instance, to create a :class:`LLSparseMatrix` (see :ref:`ll_mat`), use the following code:

..  code-block:: python

    from cysparse.sparse.ll_mat import MakeLLSparseMatrix
    
    A =  MakeLLSparseMatrix(nrow=4, ncol=3)
    
:func:`MakeLLSparseMatrix` is really a function, not a class. This not very Pythonesque approach is made necessary because :program:`Cython` doesn't allow the use of pure C variables as arguments in the constructors of classes [#use_of_pure_c_variables_in_constructors]_.

If you don't use a factory method: 

..  code-block:: python

    A = LLSparseMatrix()

you'll get the following error:

..  code-block:: bash

    AssertionError: Matrix must be instantiated with a factory method
    
..  warning::  An ``LLSparseMatrix`` can **only** be instantiated through a factory method.


Helpers
--------

``size``
""""""""""

``size`` is **not** an attribute... 

..  raw:: html

    <h4>Footnotes</h4>

..  [#factory_method_strange_name] The term *factory method* is coined by the Design Pattern community. The *method* in itself can be a function, method, class, ...
    
..  [#use_of_pure_c_variables_in_constructors] This not exactly true. :program:`Cython` allows to pass some pure C variables that can be *easily* mapped to :program:`Python` arguments. The idea is that the same arguments are 
    passed to ``__cinit__()`` **and** ``__init__()`` methods.    

