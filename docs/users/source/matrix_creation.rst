
..  _matrix_creation:

========================
How to create a matrix?
========================

To create a matrix, we use *factory methods* [#factory_method_strange_name]_: 
functions that return an object corresponding
to their arguments. Different arguments make them return different kind of objects (matrices).

..  warning::  An sparse matrix can **only** be instantiated through a factory method.

Before you can use any type of sparse matrix, you **must** first instantiate an ``LLSparseMatrix``. This matrix is well suited for construction but is not very optimized for most matrix operations. 
Once you have an ``LLSparseMatrix``, you can create a specialized sparse matrix from it.

..  warning:: The first matrix you are allowed to instantiate can **only** be an ``LLSparseMatrix``.

So basically, you create an ``LLSparseMatrix``, populate it by for instance add and/or delete elements, rows, columns, sub-matrices at will. Once you have constructed your matrix, it is time to transform it into an appropriate 
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

A ``CSCSparseMatrix`` can also be transformed into a ``CSRSparseMatrix`` for instance:

..  code-block:: python

    # transform a CSCSparseMatrix into a CSRSparseMatrix
    D = C.to_csr()

But the first matrix **must** be an ``LLSparseMatrix``.
    
Before we look how to create any matrix like object from an ``LLSparseMatrix`` or other sparse objects, let's look at the different ways to create the first ``LLSparseMatrix`` in the next section.


List of ``LLSparseMatrix`` factory methods
===========================================


``LLSparseMatrix``
----------------------

This is the main basic factory method to create an :class:`LLSparseMatrix`. There are basically **three** ways to use it:

- From specifications. Use ``nrow`` and ``ncol`` or ``size`` to specify the dimension of the new matrix. You can also provide a
  ``size_hint`` argument to (pre)allocate some space for the elements in advance.
  
  ..  code-block:: python

      A = LLSparseMatrix(nrow=256, ncol=3398, size_hint=600)

  which returns an empty ``LLSparseMatrix [INT64_t, FLOAT64_t] of size=(256, 3398) with 0 non zero values <Storage scheme: General and without zeros>`` matrix that is ready to hold 600 elements of type ``FLOAT64_t``.
  Notice that neither ``size`` nor ``size_hint`` are attributes.
  
  You can change the index type and/or the element type:
  
  ..  code-block:: python

      A = LLSparseMatrix(size=5578, size_hint=600, itype=INT32_T, dtype=COMPLEX128_T)
      
  which returns a corresponding ``LLSparseMatrix [INT32_t, COMPLEX128_t] of size=(5578, 5578) with 0 non zero values <Storage scheme: General and without zeros>`` matrix.
  
  The ``itype`` and ``dtype`` arguments represent respectively the index and element types. By default, the index and element types are (``INT32_t``, ``FLOAT32_T``) and (``INT64_t``, ``FLOAT64_T``) on 32bits and 64bits 
  platforms [#modify_default_types_in_cysparse_cfg]_. 
  
           
``LLSparseMatrixFromMMFile``
-----------------------------


``DiagonalLLSparseMatrix``
------------------------------

``IdentityLLSparseMatrix``
------------------------------

``ArrowheadLLSparseMatrix``
-------------------------------

``LinearFillLLSparseMatrix``
------------------------------

``PermutationLLSparseMatrix``
-------------------------------



..  raw:: html

    <h4>Footnotes</h4>

..  [#factory_method_strange_name] The term *factory method* is coined by the Design Pattern community. The *method* in itself can be a function, method, class, ... In :program:`CySparse`, we use global functions.
    This not very Pythonesque approach is made necessary because :program:`Cython` doesn't allow the use of pure C variables as arguments in the constructors of classes [#use_of_pure_c_variables_in_constructors]_.
    
..  [#use_of_pure_c_variables_in_constructors] This not exactly true. :program:`Cython` allows to pass some pure C variables that can be *easily* mapped to :program:`Python` arguments. The idea is that the same arguments are 
    passed to ``__cinit__()`` **and** ``__init__()`` methods. 
    
..  [#modify_default_types_in_cysparse_cfg] You can change this default behavior by giving other values in the :file:`cysparse.cfg` configuration file. See :ref:`cysparse_configuration_file`.   

