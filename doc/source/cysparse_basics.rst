..  _cysparse_basics:

=========================================================
:program:`CySparse`\'s basics
=========================================================

In this section, we explain the very basics of :program:`CySparse`. Most matrices share some common basic attributes that we detail here. To create a matrix, we use *factory methods* [#factory_method_strange_name]_: 
functions that return an object corresponding
to their arguments. Different arguments make them return different kind of objects (matrices).

Common attributes
==================

Most attributes and **all** common ones are *read-only*. Some attributes ask for some expensive computation. We indicate whenever this is the case.
Some attributes can also be given as arguments to most factory methods. We also detail which ones. 

``nrow`` and ``ncol``
----------------------

``nrow`` and ``ncol`` give respectively the number of rows and columns. You also can grab both at the same time with the ``shape`` attribute:

..  code-block:: python

    A = ...
    A.shape == A.nrow, A.ncol  # is True
    
You can use ``nrow`` and ``ncol`` as arguments to construct a new matrix. Whenever the number of rows is equal to the number of columns, i.e. when the matrix is square, you can
instead use the argument ``size=...`` in most factory methods.

``dtype`` and ``itype``
-------------------------

Each matrix (matrix-like) object has an internal index *type* and stores *typed* elements. Both types (enums) can be retrieved.
``dtype`` returns the type of the elements of the matrix and ``itype`` returns its index type.
 
See section :ref:`availabe_types` about the available types.

``use_symmetric_storage``
----------------------------

Symmetric matrices can be stored by only storing the **lower** triangular part and the diagonal of a matrix. To create a symmetric matrix, add the arguement ``use_symmetric_storage=True`` to the call of one of the factory methods.
The attribute ``use_symmetric_storage`` returns if this storage method is used or not. Thus, if ``use_symmetric_storage`` is ``True``, you know that you deal with a symmetric matrix **and** that roughly only half of its elements are stored. If 
``use_symmetric_storage`` is ``False``, it simply means that this storage scheme is not used. The matrix itself migth be symmetric or not.

``use_zero_storage``
------------------------------

By default non specified (implicit) elements are zero (``0``, ``0.0`` or ``0+0j``). :program:`CySparse` allow the user to store explicitely zeros. To explicitely store zeros, declare ``use_zero_storage=True`` as an argument
in any factory method:

..  code-block:: python

    A = NewLLSparseMatrix(use_zero_storage=True, ...)
    
The matrix ``A`` will store any zero explicitely as will any matrix created from it. You can access the value of this attribute:

..  code-block:: python

    A.use_zero_storage
    
returns ``True`` for our example. This attribute is read-only and cannot be changed. If you want to temporarily exclude zeros in some operations, you can use the ``NonZeros`` context manager:

..  code-block:: python

    with NonZeros(A):
        # use some method to add entries to A but disregard zeros entries
        ...

This context manager temporarily set the ``use_zero_storage`` attribute to ``False`` before restoring its inital value.

By default, ``use_zero_storage`` is set to ``False``.



``is_mutable``
--------------------

``is_mutable`` returns if the matrix can be modified or not. Note that for the moment, **only** an :class:`LLSparseMatrix` matrix can be modified.

``type`` and ``type_name``
-----------------------------

Each matrix or matrix-like object has its own type and type name defined as strings. For instance:

..  code-block:: python

    A = NewLLSparseMatrix(size=10, dtype=COMPLEX64_T, itype=INT32_T)
    print A.type
    print A.type_name
    
returns

..  code-block:: bash

    LLSparseMatrix
    LLSparseMatrix [INT32_t, COMPLEX64_t]

The type ``LLSparseMatrix`` is common among ``LL`` sparse format matrices while the ``type_name`` gives the specific details of the index and element types.

``nnz``
---------

The ``nnz`` attribute returns the number of "non zeros" stored in the matrix. Notice that ``0`` could be stored if ``use_zero_storage`` is set to ``True`` and if so, it will be counted in the number of "non zero" elements.
Whenever the symmetric storage scheme is used (``use_symmetric_storage`` is ``True``), ``nnz`` only returns the number of "non zero" elements stored in the lower triangular part and the diagonal of the matrix, i.e. ``nnz`` 
returns exactly how many elements are stored internally.

..  warning:: ``nnz`` returns the number of elements stored internally.

When using views, this attribute is **costly** to retrieve as it is systematically recomputed each time and we don't make any assomption on the views (views can represent matrices with rows and columns in any order and duplicated 
rows and columns any number of times). The number returned is the number of "non zero" elements stored in the equivalent matrix using the **same** storage scheme than viewed matrix.
    

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

    <h4>Footnotes</h4>

..  [#factory_method_strange_name] The term *factory method* is coined by the Design Pattern community. The *method* in itself can be a function, method, class, ...
    
..  [#use_of_pure_c_variables_in_constructors] This not exactly true. :program:`Cython` allows to pass some pure C variables that can be *easily* mapped to :program:`Python` arguments. The idea is that the same arguments are 
    passed to ``__cinit__()`` **and** ``__init__()`` methods.    

