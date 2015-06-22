..  _cysparse_basics:

=========================================================
:program:`CySparse`\'s basics
=========================================================

In this section, we expose the basics of :program:`CySparse`: how to create a matrix, what are the common attributes, etc.

Common attributes
==================

``store_zeros``
------------------

By default non specified (implicit) elements are zero (``0``, ``0.0`` or ``0+0j``). :program:`CySparse` allow the user to store explicitely zeros. To explicitely store zeros, declare ``store_zeros=True`` as an argument
in any factory method:

..  code-block:: python

    A = NewLLSparseMatrix(store_zeros=True, ...)
    
The matrix ``A`` will store any zero explicitely as will any matrix created from it. You can access the value of this attribute:

..  code-block:: python

    A.store_zeros
    
returns ``True`` for our example. This attribute is read-only and cannot be changed. If you want to temporarily exclude zeros in some operations, you can use the ``NonZeros`` context manager:

..  code-block:: python

    with NonZeros(A):
        # use some method to add entries to A but disregard zeros entries
        ...

This context manager temporarily set the ``store_zeros`` attribute to ``False`` before restoring its inital value.

By default, ``store_zeros`` is set to ``False``.
    
..  topic:: Factory method or factory function?
    
    We use **functions** to create ``LLSparseMatrix`` matrices, so why do we speak about factory **methods**? It is simply because in programming pattern parlance we speak about *factory methods* in general.



How to create a matrix?
========================

Before you can use any type of sparse matrix, you **must** first instantiate an ``LLSparseMatrix``. This matrix is well suited for construction but is not very optimized for most matrix operations. Once you have an ``LLSparseMatrix``, you can create a specialized sparse matrix from it.

Sparse matrices all come from a ``LLSparseMatrix``
------------------------------------------------------

..  _matrices_must_be_instantiated_by_a_factory_method:

``LLSparseMatrix`` matrices must be instantiated by a factory method
---------------------------------------------------------------------------

Matrices **must** be instantiated by one of the factory methods. Although we talk about factory *methods*, we mean factory *functions*.
For instance, to create a (specialized) :class:`LLSparseMatrix` (see :ref:`ll_mat`), use the following code:

..  code-block:: python

    from cysparse.sparse.ll_mat import MakeLLSparseMatrix
    
    A =  MakeLLSparseMatrix(nrow=4, ncol=3)
    
:func:`MakeLLSparseMatrix` is really a function, not a class. This not very Pythonesque approach is made necessary because :program:`Cython` doesn't allow the use of pure C variables as arguments in the constructors of classes [#use_of_pure_c_variables_in_constructors]_.

If you don't use a factory method: 

..  code-block:: python

    A = m.LLSparseMatrix()

you'll get the following error:

..  code-block:: bash

    AssertionError: Matrix must be instantiated with a factory method
    
..  warning::  An ``LLSparseMatrix`` can **only** be instantiated through a factory method.


..  raw:: html

    <h4>Footnote</h4>
    
..  [#use_of_pure_c_variables_in_constructors] This not exactly true. :program:`Cython` allows to pass some pure C variables that can be *easily* mapped to :program:`Python` arguments. The idea is that the same arguments are 
    passed to ``__cinit__()`` **and** ``__init__()`` methods.    

