..  _sparse_matrix_hierarchy:

=========================================================
:program:`CySparse`\'s sparse matrix classes hierachy
=========================================================

The hierarchy at a glance
===========================


The :class:`SparseMatrix` class
=================================


..  _matrices_must_be_instantiated_by_a_factory_method:

Matrices must be instantiated by a factory method
--------------------------------------------------

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

The :class:`MutableSparseMatrix` class
=======================================

The :class:`ImmutableSparseMatrix` class
=========================================


..  raw:: html

    <h4>Footnote</h4>
    
..  [#use_of_pure_c_variables_in_constructors] This not exactly true. :program:`Cython` allows to pass some pure C variables that can be *easily* mapped to :program:`Python` arguments. The idea is that the same arguments are 
    passed to ``__cinit__()`` **and** ``__init__()`` methods.    

