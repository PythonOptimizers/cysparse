..  cysparse_for_cython_users:

=========================================================
:program:`CySparse` for :program:`Cython` users
=========================================================

I efficiency is a major concern to you, we strongly encourage you to use :program:`Cython` to 
compile your own Python extension.

How to compile with :program:`Cython`?
======================================

To compile and link your project with :program:`CySparse`, simply download the complete source code of :program:`CySparse` and refer to the ``.pxd`` files as needed. 

Creation of matrices
====================

Matrices cannot be instantiated directly in :program:`Python` (see :ref:`matrices_must_be_instantiated_by_a_factory_method`). In :program:`Cython`, the factory methods
can be used **except** if this creates a *circular dependencies* between these methods. One solution is to simply invoke
the protection mechanism for the creation of classes yourself:

..  code-block::  cython

    from cysparse.cysparse.sparse_mat cimport unexposed_value
    from cysparse.cysparse.ll_mat cimport LLSparseMatrix
    
    LLSparseMatrix ll_mat = LLSparseMatrix(control_object=unexposed_value, ...)

By adding ``control_object=unexposed_value`` as argument, the ``ll_mat`` assertion in the ``__cinit__()`` constructor will be not be triered. Be carreful though as you are fully responsible 
for the creation of the matrix.
 
Accessing matrix elements
==========================
