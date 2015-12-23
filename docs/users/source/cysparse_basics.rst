..  _cysparse_basics:

=========================================================
:program:`CySparse`\'s basics
=========================================================

In this section, we explain the very basics of :program:`CySparse`. Most matrices share some common basic attributes that we detail here. 

Most attributes and **all** common ones are *read-only*. Some attributes ask for some expensive computation. We indicate whenever this is the case.
Some attributes can also be given as arguments to most factory methods. We also detail which ones. 

Common strorage attributes
===========================

For efficiency reasons, :program:`CySparse` can use different storage schemes/methods to store matrices. For instance, symmetric matrices can be stored with only half of their elements. 

``store_symmetric``
----------------------------

Symmetric matrices can be stored by only storing the **lower** triangular part and the diagonal of a matrix. To create a symmetric matrix, add the arguement ``store_symmetric=True`` to the call of one of the factory methods.
The attribute ``store_symmetric`` returns if this storage method is used or not. Thus, if ``store_symmetric`` is ``True``, you know that you deal with a symmetric matrix **and** that roughly only half of its elements are stored. If 
``store_symmetric`` is ``False``, it simply means that this storage scheme is not used. The matrix itself migth be symmetric or not.

``store_zero``
------------------------------

By default non specified (implicit) elements are zero (``0``, ``0.0`` or ``0+0j``). :program:`CySparse` allow the user to store explicitely zeros. To explicitely store zeros, declare ``store_zero=True`` as an argument
in any factory method:

..  code-block:: python

    A = LLSparseMatrix(store_zero=True, ...)
    
The matrix ``A`` will store any zero explicitely as will any matrix created from it. You can access the value of this attribute:

..  code-block:: python

    A.store_zero
    
returns ``True`` for our example. This attribute is read-only and cannot be changed. If you want to temporarily exclude zeros in some operations, you can use the ``NonZeros`` context manager:

..  code-block:: python

    with NonZeros(A):
        # use some method to add entries to A but disregard zeros entries
        ...

This context manager temporarily set the ``store_zero`` attribute to ``False`` before restoring its inital value.

By default, ``store_zero`` is set to ``False``.

``is_mutable``
--------------------

``is_mutable`` returns if the matrix can be modified or not. Note that for the moment, **only** an :class:`LLSparseMatrix` matrix can be modified.


Common content attributes
=========================


``nrow`` and ``ncol``
----------------------

``nrow`` and ``ncol`` give respectively the number of rows and columns. You also can grab both at the same time with the ``shape`` attribute:

..  code-block:: python

    A = ...
    A.shape == A.nrow, A.ncol  # is True
    
You can use ``nrow`` and ``ncol`` as arguments to construct a new matrix. Whenever the number of rows is equal to the number of columns, i.e. when the matrix is square, you can
instead use the argument ``size=...`` in most factory methods.

``nnz``
---------

The ``nnz`` attribute returns the number of "non zeros" stored in the matrix. Notice that ``0`` could be stored if ``store_zero`` is set to ``True`` and if so, it will be counted in the number of "non zero" elements.
Whenever the symmetric storage scheme is used (``store_symmetric`` is ``True``), ``nnz`` only returns the number of "non zero" elements stored in the lower triangular part and the diagonal of the matrix, i.e. ``nnz`` 
returns exactly how many elements are stored internally.

..  warning:: ``nnz`` returns **exactly** the number of elements stored internally.

When using views, this attribute is **costly** to retrieve as it is systematically recomputed each time and we don't make any assomption on the views (views can represent matrices with rows and columns in any order and duplicated 
rows and columns any number of times). The number returned is the number of "non zero" elements stored in the equivalent matrix using the **same** storage scheme than viewed matrix.
    


Common type attributes
=========================

``dtype`` and ``itype``
-------------------------

Each matrix (matrix-like) object has an internal index *type* and stores *typed* elements. Both types (enums) can be retrieved.
``dtype`` returns the type of the elements of the matrix and ``itype`` returns its index type.
 
See section :ref:`availabe_types` about the available types.

``is_symmetric``
-------------------

[TODO in the code!!!]

Returns if the matrix is symmetric or not. While matrices using the symmetric storage (``store_symmetric == True``) are symmetric by definition and ``is_symmetric`` returns immediatly ``True``, this attribute is costly to 
compute in general.



Common string attributes
===========================

Some attributes are stored as ``C`` struct internally and can thus not be accessed from :program:`Python`. We do however provide some strings for the most important ones.

``base_type_str`` and ``full_type_str``
------------------------------------------

Each matrix or matrix-like object has its own type and type name defined as strings. For instance:

..  code-block:: python

    A = NewLLSparseMatrix(size=10, dtype=COMPLEX64_T, itype=INT32_T)
    print A.base_type_str
    print A.full_type_str
    
returns

..  code-block:: bash

    LLSparseMatrix
    LLSparseMatrix [INT32_t, COMPLEX64_t]

The type ``LLSparseMatrix`` is common among ``LL`` sparse format matrices while the ``full_type_str`` gives the specific details of the index and element types.



Typed matrices in :program:`Python`?
======================================





